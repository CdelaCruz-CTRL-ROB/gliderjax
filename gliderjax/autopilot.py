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

roll_control is the author's roll law (nonlinear PID with gravity
compensation; the d_phi, m_phi and Sigma_psi terms are not in yet).
yaw_control is the author's heading PID (roll as virtual control).
pitch_control is the placeholder, accepted by the author as equivalent to
the author's own nonlinear PID. Each one
gets N_MEM numbers of memory (integrators, previous values, filter states)
carried between calls, and the reference together with its rate
(ref_dot, the backward difference of the reference over one controller
period).

ctrl_state layout, all scope-able via tel["ctrl_state"] (scope "ctrl_state:i"):
  [0:4]    pitch memory    ([0] = pitch PID integral term)
  [4:8]    roll memory     ([0] = roll PID integral term)
  [8:12]   yaw memory      ([0] = yaw PID integral term)
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
KP_P, KD_P, KI_P = 0.015, 0.08, 0.005       # pitch PID (x_g units / rad)
I_MAX_P = 0.005                              # [m of x_g] max |integral term|
KP_Y, KD_Y, KI_Y = 0.35, 0.2, 0.003          # yaw PID (rad of roll / rad)
I_MAX_Y = 0.1                                # [rad of roll] max |integral term|
ROLL_MAX = 0.8                              # [rad] roll_ref saturation
KP_R, KD_R, KI_R = 0.008, 0.02, 0.0008          # roll PID (m of u / rad);
#                                              tuned by the author (100 Hz)
I_MAX_R = 0.001                              # [m of u] max |integral term|
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


def nominal_mass(params):
    """(m_tot, z_g0) of the known masses, as the controllers see them:
    no m_dist, rotating mass at gamma = 0."""
    m_tot = params.m_static + params.m_lin + params.m_rot + params.m_pist
    zg0 = (params.m_static * params.r_static[2] + params.m_lin * params.r_lin0[2]
           + params.m_rot * params.R_rot) / m_tot
    return m_tot, zg0


def pitch_control(x, params, pitch_ref, pitch_ref_dot, mem, dt):
    """PLACEHOLDER pitch PID -> moving mass. Returns (rc_cmd [m], mem).
    mem[0] = integral term KI_P * int(e) [m of x_g], limited to +-I_MAX_P."""
    th = x[4]
    th_dot = euler_rates(x)[1]
    m_tot, zg0 = nominal_mass(params)     # for the gravity feedforward
    e = th - pitch_ref
    de = th_dot - pitch_ref_dot
    integ = jnp.clip(mem[0] + KI_P * e * dt, -I_MAX_P, I_MAX_P)
    xg_cmd = (1.0 / jnp.cos(th)) * (KP_P * e + KD_P * de + integ
                                    - zg0 * jnp.sin(th))
    rc_cmd = m_tot * xg_cmd / params.m_lin
    return rc_cmd, mem.at[0].set(integ)


def roll_control(x, params, roll_ref, roll_ref_dot, mem, dt):
    """Author's roll law, nonlinear PID with gravity compensation part
    (without the -d_phi/W phi_dot, -m_phi/W phi_dd_d and Sigma_psi terms):
        u = (1/cos phi) (KP e + KD e_dot + KI int(e) - z_g sin phi)
    -> rotating mass. Returns (gamma_cmd [rad], mem).
    mem[0] = integral term KI_R * int(e) [m of u], limited to +-I_MAX_R.
    u is the design's "y_g": lateral CG shift to PORT, i.e. -y_g of the body
    frame (the way the rotating mass moves for gamma > 0). With it roll has
    the same form as pitch, M_x = -W cos(th) (z_g sin(phi) + u cos(phi)),
    and e = measured - ref works with positive gains. See docs/DISENO_ROLL.md."""
    phi = x[3]
    phi_dot = euler_rates(x)[0]          # NOT p (body rate)
    m_tot, zg0 = nominal_mass(params)
    e = phi - roll_ref
    de = phi_dot - roll_ref_dot
    integ = jnp.clip(mem[0] + KI_R * e * dt, -I_MAX_R, I_MAX_R)
    u = (1.0 / jnp.cos(phi)) * (KP_R * e + KD_R * de + integ
                                - zg0 * jnp.sin(phi))
    # u -> gamma: y_rot = -R_rot sin(gamma), so u = -y_g = m_rot R_rot sin(gamma) / m_tot
    sin_gamma = jnp.clip(u * m_tot / (params.m_rot * params.R_rot), -1.0, 1.0)
    return jnp.arcsin(sin_gamma), mem.at[0].set(integ)


def yaw_control(x, params, yaw_ref, yaw_ref_dot, turn_sign, mem, dt):
    """Author's heading law -> roll reference (virtual control):
        roll_ref = (-KP e - KD e_dot - KI int(e)) * turn_sign,
        e = wrap(psi - psi_ref)   (measured - ref, the short way round)
    turn_sign = roll-yaw coupling sign (rollyawdir): +1 descent, -1 climb,
    0 = no heading control. Returns (roll_ref [rad], mem), limited to
    +-ROLL_MAX. mem[0] = integral term KI_Y * int(e) [rad of roll], limited
    to +-I_MAX_Y."""
    e = wrap(x[5] - yaw_ref)
    psi_dot = euler_rates(x)[2]          # NOT r (body rate)
    de = psi_dot - yaw_ref_dot
    integ = jnp.clip(mem[0] + KI_Y * e * dt, -I_MAX_Y, I_MAX_Y)
    roll_ref = (-KP_Y * e - KD_Y * de - integ) * turn_sign
    return jnp.clip(roll_ref, -ROLL_MAX, ROLL_MAX), mem.at[0].set(integ)


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
