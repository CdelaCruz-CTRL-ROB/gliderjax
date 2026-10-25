"""Autopilot: attitude cascade + mission-mode controller factories.

Three modes, one architecture (all compatible with make_simulator):

  MANUAL          your own controller fn returning raw actuator cmds
                  (run_test.py unchanged -- nothing here needed), or
                  make_open_loop_controller(schedule): piecewise-constant
                  commands from a table of phases
  SEMI-AUTOMATIC  make_attitude_controller(refs_fn, roll_from_yaw):
                  refs_fn(t) -> [pitch_ref, yaw_ref, roll_ref, dB_cmd, turn_sign]
                  attitude loops + reference shaping, NO state machine.
                  roll_from_yaw=False: pitch and roll loops on their own
                  (roll_ref from the table); True: yaw loop -> roll_ref
  WAYPOINT        make_waypoint_controller(mission):
                  full V2 FSM -> reference shaping -> attitude loops

THE THREE LOOPS ARE PLACEHOLDERS (working, but NOT your Simulink
Pitch_control / Roll control / Yaw_control blocks): pitch_control,
roll_control and yaw_control below are the functions to replace. Each one
gets N_MEM numbers of memory (integrators, previous values, filter states)
carried between calls, and the reference together with its rate
(ref_dot, the backward difference of the reference over one controller
period).

ctrl_state layout, all scope-able via tel["ctrl_state"] (scope "ctrl_state:i"):
  [0:4]    pitch memory    ([0] = placeholder pitch PID integrator)
  [4:8]    roll memory
  [8:12]   yaw memory
  [12:15]  references the loops used [pitch, roll, yaw] [rad] (also the
           previous value for ref_dot)
  [15:24]  reference shaping states (3x3)
  waypoint mode only:
  [24:38]  FSM carry (see navigation.py)
  [38:40]  held GPS fix [lon_deg, lat_deg]
"""
import jax.numpy as jnp

from .params import G0
from .dynamics import T_euler
from .sim import make_ref_pipeline
from .navigation import build_fsm, gps_fix
from .geodesy import ecef2lla

# ---- placeholder attitude gains (tuned for the CIDESI soft-z_g vehicle) ----
KP_P, KD_P, KI_P = 0.015, 0.08, 0.003        # pitch PID (x_g units / rad)
I_MAX_P = 0.005                              # [m of x_g] max |integral term|
K_YAW = 1.2                                  # yaw err -> roll_ref [rad/rad]
ROLL_MAX = 0.45                              # [rad] roll_ref saturation
K_ROLL = 2.0                                 # roll err -> gamma [rad/rad]
BUOY_SIGN = -1.0
# BUOY_SIGN maps the FSM's V2 convention (Buoy_ref: +3 dive, -3 climb) to the
# VBD block's physical dB_ref (negative dB -> less buoyant -> dive).
# <-- CONFIRM against the Simulink wiring between the FSM and Buoyancy block.


def wrap(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


# ----------------------------------------------------------------------------
# The three loops. Each: (x, params, ref, ref_dot, mem, dt) -> (output, mem).
#   x      true state (26,): x[2] z, x[3] roll, x[4] pitch, x[5] yaw,
#          x[6:9] u v w, x[9] p, x[10] q, x[11] r   [m, rad, m/s, rad/s]
#          p q r are BODY rates, not the Euler-angle rates: use
#          euler_rates(x) for [phi_dot, theta_dot, psi_dot]
#   ref_dot  rate of the reference [rad/s] (backward difference over dt;
#          the shaped pitch/yaw refs are smooth; an unshaped roll step
#          gives one sample of (step / dt))
#   mem    (N_MEM,) this loop's memory, returned updated
#   dt     controller period [s]
# JAX rules: use jnp (not np/math), no python `if` on values (jnp.where,
# jnp.clip), keep everything that must persist in mem.
# ----------------------------------------------------------------------------
N_MEM = 4


def euler_rates(x):
    """[phi_dot, theta_dot, psi_dot] = T(phi, theta) @ [p, q, r]
    (the attitude part of eta_dot = J(eta) nu)."""
    return T_euler(x[3], x[4]) @ x[9:12]


def pitch_control(x, params, pitch_ref, pitch_ref_dot, mem, dt):
    """PLACEHOLDER pitch PID -> moving mass. Returns (rc_cmd [m], mem).
    mem[0] = integral term KI_P * int(e) [m of x_g], limited to +-I_MAX_P."""
    th = x[4]
    th_dot = euler_rates(x)[1]
    # gravity feedforward derived from the parameters
    m_tot = params.m_static + params.m_lin + params.m_rot + params.m_pist
    zg0 = (params.m_static * params.r_static[2] + params.m_lin * params.r_lin0[2]
           + params.m_rot * params.R_rot) / m_tot
    e = th - pitch_ref
    de = th_dot - pitch_ref_dot
    integ = jnp.clip(mem[0] + KI_P * e * dt, -I_MAX_P, I_MAX_P)
    xg_cmd = (1.0 / jnp.cos(th)) * (KP_P * e + KD_P * de + integ
                                    - zg0 * jnp.sin(th))
    rc_cmd = m_tot * xg_cmd / params.m_lin
    return rc_cmd, mem.at[0].set(integ)


def roll_control(x, params, roll_ref, roll_ref_dot, mem, dt):
    """PLACEHOLDER roll P -> rotating mass. Returns (gamma_cmd [rad], mem).
    Plant sign: gamma > 0 -> roll < 0, hence the minus."""
    phi = x[3]
    gamma_cmd = -K_ROLL * (roll_ref - phi)
    return gamma_cmd, mem


def yaw_control(x, params, yaw_ref, yaw_ref_dot, turn_sign, mem, dt):
    """PLACEHOLDER heading P -> roll reference. Returns (roll_ref [rad], mem).
    turn_sign = roll-to-turn sign: +1 descent, -1 climb, 0 = no heading
    control."""
    e_y = wrap(yaw_ref - x[5])
    roll_ref = jnp.clip(K_YAW * e_y, -ROLL_MAX, ROLL_MAX) * turn_sign
    return roll_ref, mem


def attitude_step(x, params, shaped, turn_sign, mem, used_prev, dt,
                  roll_from_yaw=True):
    """Wires the three loops. shaped = [pitch_ref, yaw_ref, roll_ref] after
    reference shaping; mem (3*N_MEM,) = [pitch | roll | yaw]; used_prev =
    the refs used in the previous call [pitch, roll, yaw] (for ref_dot).
    roll_from_yaw (python bool, fixed when the controller is built): the
    roll reference comes from the yaw loop (True) or from shaped[2] (False).
    Returns (cmds [rc, gamma, 0 = dB slot], mem, used refs [pitch, roll, yaw])."""
    mp, mr, my = mem[0:N_MEM], mem[N_MEM:2 * N_MEM], mem[2 * N_MEM:3 * N_MEM]
    pitch_ref_dot = (shaped[0] - used_prev[0]) / dt
    yaw_ref_dot = wrap(shaped[1] - used_prev[2]) / dt
    rc_cmd, mp = pitch_control(x, params, shaped[0], pitch_ref_dot, mp, dt)
    if roll_from_yaw:
        roll_ref, my = yaw_control(x, params, shaped[1], yaw_ref_dot,
                                   turn_sign, my, dt)
    else:
        roll_ref = shaped[2]
    roll_ref_dot = (roll_ref - used_prev[1]) / dt
    gamma_cmd, mr = roll_control(x, params, roll_ref, roll_ref_dot, mr, dt)
    used = jnp.array([shaped[0], roll_ref, shaped[1]])
    return (jnp.array([rc_cmd, gamma_cmd, 0.0]),
            jnp.concatenate([mp, mr, my]), used)


def piecewise_schedule(schedule):
    """schedule = [(t_start, v1, v2, ...), ...] sorted by t_start ->
    f(t) = values of the last row with t_start <= t (like a Signal Builder)."""
    sch = jnp.asarray(schedule, dtype=float)
    t_starts, rows = sch[:, 0], sch[:, 1:]

    def f(t):
        # +1e-6: t is accumulated as t + dt, so a phase starting at 300.0
        # must not wait one extra step because t reads 299.99999999
        i = jnp.searchsorted(t_starts, t + 1e-6, side="right") - 1
        return rows[jnp.clip(i, 0, rows.shape[0] - 1)]

    return f


def make_open_loop_controller(schedule):
    """MANUAL mode, time-scheduled (like a Simulink Signal Builder).
    schedule = [(t_start, rc_cmd [m], gamma_cmd [rad], dB_cmd [N]), ...]
    sorted by t_start; each row's commands hold until the next row starts.
    dB_cmd in PHYSICAL VBD convention (negative = dive).
    Returns (ctrl_state0, controller)."""
    cmds = piecewise_schedule(schedule)

    def controller(t, x, cs, params):
        return cmds(t), cs

    return jnp.zeros(1), controller


def make_attitude_controller(refs_fn, dt_ctrl, roll_from_yaw=True,
                             ref_pipeline_kwargs=None, init_refs=(0.0, 0.0, 0.0)):
    """SEMI-AUTOMATIC mode.
    refs_fn(t) -> [pitch_ref, yaw_ref, roll_ref, dB_cmd, turn_sign]
    (dB_cmd in PHYSICAL VBD convention: negative = dive). roll_from_yaw:
    see attitude_step (False: roll_ref is used, yaw_ref and turn_sign are
    ignored; True: the other way around).
    init_refs = (pitch, yaw, roll) [rad] where the shaped references start
    (use the initial attitude; 0 ramps from 0)."""
    kw = dict(ref_pipeline_kwargs or {})
    kw.setdefault("init", tuple(init_refs))
    ref0, ref_step = make_ref_pipeline(dt_ctrl, **kw)
    n = 3 * N_MEM

    def controller(t, x, cs, params):
        mem, used = cs[0:n], cs[n:n + 3]
        ref_s = cs[n + 3:n + 12].reshape(3, 3)
        raw = refs_fn(t)
        ref_s, shaped = ref_step(ref_s, jnp.array([raw[0], raw[1], raw[2]]))
        cmds, mem, used = attitude_step(x, params, shaped, raw[4], mem, used,
                                        dt_ctrl, roll_from_yaw)
        cmds = cmds.at[2].set(raw[3])
        return cmds, jnp.concatenate([mem, used, ref_s.ravel()])

    # previous refs start at the initial ones, so ref_dot starts at 0
    pitch0, yaw0, roll0 = kw["init"]
    used0 = jnp.array([pitch0, roll0, yaw0], dtype=float)
    return jnp.concatenate([jnp.zeros(n), used0, ref0.ravel()]), controller


def make_waypoint_controller(mission, dt_ctrl, ref_pipeline_kwargs=None):
    """WAYPOINT mode: FSM -> reference shaping -> attitude cascade.
    Returns (ctrl_state0, controller, ecef0). Initialize the sim with
    initial_state(..., ecef0=ecef0) so GPS truth starts at the origin."""
    kw = ref_pipeline_kwargs or {}
    ref0, ref_step = make_ref_pipeline(dt_ctrl, **kw)
    fsm0, ecef0, fsm_step = build_fsm(mission)
    origin_deg = jnp.array([mission[0][0], mission[0][1]])

    n = 3 * N_MEM

    def controller(t, x, cs, params):
        mem, used = cs[0:n], cs[n:n + 3]
        ref_s = cs[n + 3:n + 12].reshape(3, 3)
        fsm_c = cs[n + 12:n + 26]
        held = cs[n + 26:n + 28]
        depth = x[2]
        # GPS (gated on surfacing; holds last fix underwater)
        fix = gps_fix(params, x[23:26], depth, held)
        # FSM tick
        fsm_c, o = fsm_step(fsm_c, fix, depth, x[5], t)
        # reference shaping (pitch & yaw; roll_ref comes from the yaw loop)
        ref_s, shaped = ref_step(ref_s, jnp.array([o[0], o[1], 0.0]))
        cmds, mem, used = attitude_step(x, params, shaped, o[3], mem, used,
                                        dt_ctrl, roll_from_yaw=True)
        cmds = cmds.at[2].set(BUOY_SIGN * o[2])
        cs = jnp.concatenate([mem, used, ref_s.ravel(), fsm_c, fix])
        return cmds, cs

    cs0 = jnp.concatenate([jnp.zeros(n + 3), ref0.ravel(), fsm0, origin_deg])
    return cs0, controller, ecef0
