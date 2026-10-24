"""Autopilot: attitude cascade + mission-mode controller factories.

Three modes, one architecture (all compatible with make_simulator):

  MANUAL          your own controller fn returning raw actuator cmds
                  (run_test.py unchanged -- nothing here needed), or
                  make_open_loop_controller(schedule): piecewise-constant
                  commands from a table of phases
  SEMI-AUTOMATIC  make_attitude_controller(refs_fn):
                  refs_fn(t) -> [pitch_ref, yaw_ref, roll_ref?, dB_cmd, cg]
                  attitude loops + reference shaping, NO state machine
  WAYPOINT        make_waypoint_controller(mission):
                  full V2 FSM -> reference shaping -> attitude loops

ATTITUDE LOOPS ARE PLACEHOLDERS (clearly working, but NOT your Simulink
Pitch_control / Yaw_control / Roll control blocks -- send those bodies and
they replace attitude_step verbatim).

ctrl_state layout (waypoint mode), all scope-able via tel["ctrl_state"]:
  [0]     pitch PID integrator
  [1:10]  reference pipeline states (3x3)
  [10:24] FSM carry (see navigation.py)
  [24:26] held GPS fix [lon_deg, lat_deg]
"""
import jax.numpy as jnp

from .params import G0
from .sim import make_ref_pipeline
from .navigation import build_fsm, gps_fix
from .geodesy import ecef2lla

# ---- placeholder attitude gains (tuned for the CIDESI soft-z_g vehicle) ----
KP_P, KD_P, KI_P = 0.015, 0.08, 0.003        # pitch PID (x_g units / rad)
K_YAW = 1.2                                  # yaw err -> roll_ref [rad/rad]
ROLL_MAX = 0.45                              # [rad] roll_ref saturation
K_ROLL = 2.0                                 # roll err -> gamma [rad/rad]
BUOY_SIGN = -1.0
# BUOY_SIGN maps the FSM's V2 convention (Buoy_ref: +3 dive, -3 climb) to the
# VBD block's physical dB_ref (negative dB -> less buoyant -> dive).
# <-- CONFIRM against the Simulink wiring between the FSM and Buoyancy block.


def wrap(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def attitude_step(x, params, refs, cg, integ, dt_ctrl):
    """PLACEHOLDER attitude cascade. refs = [pitch_ref, yaw_ref] (shaped).
    Returns (cmds[rc, gamma, dB=0 slot], new_integ, roll_ref)."""
    th, q = x[4], x[10]
    phi = x[3]
    yaw = x[5]
    # ---- pitch PID with parameter-derived gravity feedforward
    m_tot = params.m_static + params.m_lin + params.m_rot + params.m_pist
    zg0 = (params.m_static * params.r_static[2] + params.m_lin * params.r_lin0[2]
           + params.m_rot * params.R_rot) / m_tot
    e = th - refs[0]
    xg_cmd = (1.0 / jnp.cos(th)) * (KP_P * e + KD_P * q + KI_P * integ
                                    - zg0 * jnp.sin(th))
    rc_cmd = m_tot * xg_cmd / params.m_lin
    # anti-windup: freeze only when the error pushes FURTHER into saturation
    # (freezing on saturation alone latches wound-up integrals -- found the
    # hard way: -74 deg dives with the mass railed)
    sat = jnp.abs(rc_cmd) >= params.dx_lin_max
    winding_in = sat & (e * rc_cmd > 0)
    integ = jnp.where(winding_in, integ, integ + e * dt_ctrl)
    # ---- yaw -> roll_ref (x coupling gain: +1 descent, -1 climb, 0 off)
    e_y = wrap(refs[1] - yaw)
    roll_ref = jnp.clip(K_YAW * e_y, -ROLL_MAX, ROLL_MAX) * cg
    # ---- roll -> gamma (plant sign: gamma>0 -> phi<0, hence the minus)
    gamma_cmd = -K_ROLL * (roll_ref - phi)
    return jnp.array([rc_cmd, gamma_cmd, 0.0]), integ, roll_ref


def make_open_loop_controller(schedule):
    """MANUAL mode, time-scheduled (like a Simulink Signal Builder).
    schedule = [(t_start, rc_cmd [m], gamma_cmd [rad], dB_cmd [N]), ...]
    sorted by t_start; each row's commands hold until the next row starts.
    dB_cmd in PHYSICAL VBD convention (negative = dive).
    Returns (ctrl_state0, controller)."""
    sch = jnp.asarray(schedule, dtype=float)
    t_starts, rows = sch[:, 0], sch[:, 1:4]

    def controller(t, x, cs, params):
        # +1e-6: t is accumulated as t + dt, so a phase starting at 300.0
        # must not wait one extra step because t reads 299.99999999
        i = jnp.searchsorted(t_starts, t + 1e-6, side="right") - 1
        return rows[jnp.clip(i, 0, rows.shape[0] - 1)], cs

    return jnp.zeros(1), controller


def make_attitude_controller(refs_fn, dt_ctrl, ref_pipeline_kwargs=None):
    """SEMI-AUTOMATIC mode. refs_fn(t) -> [pitch_ref, yaw_ref, dB_cmd, cg]
    (dB_cmd in PHYSICAL VBD convention: negative = dive)."""
    kw = ref_pipeline_kwargs or {}
    ref0, ref_step = make_ref_pipeline(dt_ctrl, **kw)

    def controller(t, x, cs, params):
        integ, ref_s = cs[0], cs[1:10].reshape(3, 3)
        raw = refs_fn(t)
        ref_s, shaped = ref_step(ref_s, jnp.array([raw[0], raw[1], 0.0]))
        cmds, integ, _ = attitude_step(x, params, shaped, raw[3], integ, dt_ctrl)
        cmds = cmds.at[2].set(raw[2])
        return cmds, jnp.concatenate([jnp.array([integ]), ref_s.ravel()])

    return jnp.zeros(10), controller


def make_waypoint_controller(mission, dt_ctrl, ref_pipeline_kwargs=None):
    """WAYPOINT mode: FSM -> reference shaping -> attitude cascade.
    Returns (ctrl_state0, controller, ecef0). Initialize the sim with
    initial_state(..., ecef0=ecef0) so GPS truth starts at the origin."""
    kw = ref_pipeline_kwargs or {}
    ref0, ref_step = make_ref_pipeline(dt_ctrl, **kw)
    fsm0, ecef0, fsm_step = build_fsm(mission)
    origin_deg = jnp.array([mission[0][0], mission[0][1]])

    def controller(t, x, cs, params):
        integ = cs[0]
        ref_s = cs[1:10].reshape(3, 3)
        fsm_c = cs[10:24]
        held = cs[24:26]
        depth = x[2]
        # GPS (gated on surfacing; holds last fix underwater)
        fix = gps_fix(params, x[23:26], depth, held)
        # FSM tick
        fsm_c, o = fsm_step(fsm_c, fix, depth, x[5], t)
        # reference shaping (pitch & yaw; roll shaped downstream of yaw loop)
        ref_s, shaped = ref_step(ref_s, jnp.array([o[0], o[1], 0.0]))
        cmds, integ, roll_ref = attitude_step(x, params, shaped, o[3],
                                              integ, dt_ctrl)
        cmds = cmds.at[2].set(BUOY_SIGN * o[2])
        cs = jnp.concatenate([jnp.array([integ]), ref_s.ravel(), fsm_c, fix])
        return cmds, cs

    cs0 = jnp.concatenate([jnp.zeros(1), jnp.zeros(9), fsm0, origin_deg])
    return cs0, controller, ecef0
