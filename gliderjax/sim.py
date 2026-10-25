"""Simulation engine: RK4 + lax.scan, discrete controllers, telemetry.

Structure (the Simulink-equivalent architecture):

  outer lax.scan over FRAMES  (one telemetry emission per frame)
      inner lax.fori_loop over `decimate` solver steps
          - controller runs every `ctrl_every` inner steps (discrete rate!)
          - actuators move toward commands with RATE + TRAVEL limits
          - continuous state integrated with a fixed-step solver
            (SOLVERS: "ode1" Euler, "ode2" Heun, "ode4" RK4 = Simulink names)

The controller is a pure function:
    cmds, ctrl_state = controller(t, x, ctrl_state, params)
with cmds = [dx_lin_cmd, gamma_cmd, dB_cmd]. Because it runs at a fixed
discrete rate inside the loop, it IS the embedded implementation.
"""
from __future__ import annotations

import functools
from typing import Callable, Any

import jax
import jax.numpy as jnp

from .params import GliderParams, ModelConfig
from .dynamics import dynamics


def rk4_step(x, dt, p, cfg):
    f = lambda y: dynamics(y, p, cfg)[0]
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(x, dt, p, cfg):
    return x + dt * dynamics(x, p, cfg)[0]


def heun_step(x, dt, p, cfg):
    f = lambda y: dynamics(y, p, cfg)[0]
    k1 = f(x)
    k2 = f(x + dt * k1)
    return x + (dt / 2.0) * (k1 + k2)


# fixed-step solvers, named as in Simulink
SOLVERS = {"ode1": euler_step, "ode2": heun_step, "ode4": rk4_step}


def _rl(prev, u, rate, dt):
    return prev + jnp.clip(u - prev, -rate * dt, rate * dt)


def move_actuators(x, cmds, dt, p: GliderParams):
    """cmds = [rc_cmd [m], gamma_cmd [rad], dB_ref [N]].
    Each channel: prefilter -> rate limiter -> saturation -> main TF.
    All TF stages are exact ZOH of the continuous num/den in params."""
    # ---- moving mass: sat -> rate limit (0.008 m/s) -> 400/(s^2+30s+400)
    m_rl = _rl(x[12], jnp.clip(cmds[0], -p.dx_lin_max, p.dx_lin_max),
               p.mass_rate, dt)
    m_ss = p.mass_A @ x[13:15] + p.mass_B * m_rl
    # ---- rotary: 1/(0.1s+1) -> rate limit -> sat(gamma_max) -> 2nd-order
    g_pre = p.rotpre_a * x[15] + p.rotpre_b * cmds[1]
    y_pre = p.rotpre_c * g_pre
    g_rl = _rl(x[16], y_pre, p.rot_rate, dt)
    g_sat = jnp.clip(g_rl, -p.gamma_max, p.gamma_max)
    g_ss = p.rot_A @ x[17:19] + p.rot_B * g_sat
    # ---- VBD: 1/(0.2s+1) -> rate limit [N/s] -> main TF -> (static map
    #      to piston happens in dynamics; stroke saturation lives there)
    v_pre = p.vbdpre_a * x[19] + p.vbdpre_b * cmds[2]
    v_rl = _rl(x[20], p.vbdpre_c * v_pre, p.vbd_rate_N, dt)
    v_ss = p.vbd_A @ x[21:23] + p.vbd_B * v_rl
    x = x.at[12].set(m_rl)
    x = jax.lax.dynamic_update_slice(x, m_ss, (13,))
    x = x.at[15].set(g_pre).at[16].set(g_rl)
    x = jax.lax.dynamic_update_slice(x, g_ss, (17,))
    x = x.at[19].set(v_pre).at[20].set(v_rl)
    x = jax.lax.dynamic_update_slice(x, v_ss, (21,))
    return x


def make_ref_pipeline(dt,
                      pitch_rate=0.1,  pitch_tf=([1.0], [5.0, 1.0]),
                      yaw_rate=0.03,   yaw_tf=([1.0], [5.0, 1.0]),
                      roll_rate=0.1,   roll_tf=([1.0], [5.0, 1.0]),
                      bypass_roll=True, bypass_pitch=False, bypass_yaw=False,
                      bypass_all=False, init=(0.0, 0.0, 0.0)):
    """REFERENCE shaping (distinct from the actuator chains!):
        raw ref -> [rate limiter] -> [main TF] -> shaped ref
    Simulink values: pitch +-0.1 rad/s -> 1/(5s+1); yaw +-0.03 rad/s ->
    1/(5s+1); roll: none in the current Simulink (bypass_roll=True default),
    but the same chain as pitch is available -- set bypass_roll=False.
    No prefilters here: the discrete rate limiter needs none.

    bypass_* / bypass_all: stages become passthrough (built with rate=1e9
    and a 1e-9 s pole; introduces one dt sample of delay, nothing else) --
    use bypass_all=True when feeding pre-designed smooth trajectories.

    All arguments are CONTINUOUS-time (rad/s rates, num/den TFs); the exact
    discrete forms are computed once here at build time.
    init = (pitch, yaw, roll) [rad]: the returned state0 is already settled
    at these values (e.g. the initial attitude), so the shaped references
    start there instead of ramping from 0.

    Returns (state0, step):
        new_state, [pitch_s, yaw_s, roll_s] = step(state, [pitch, yaw, roll])
    Carry `state` inside your controller's ctrl_state.
    """
    from .params import zoh_tf
    PASS_TF, PASS_R = ([1.0], [1e-9, 1.0]), 1e9

    def build(rate, tf, bypass):
        if bypass or bypass_all:
            rate, tf = PASS_R, PASS_TF
        A, B, C = zoh_tf(*tf, dt, 2)
        return rate, A, B, C

    chans = [build(pitch_rate, pitch_tf, bypass_pitch),
             build(yaw_rate, yaw_tf, bypass_yaw),
             build(roll_rate, roll_tf, bypass_roll)]
    # per channel: [rl, main0, main1], settled at init (zeros if init = 0)
    state0 = jnp.stack([
        jnp.concatenate([jnp.array([v]),
                         jnp.linalg.solve(jnp.eye(2) - A, B * v)])
        for v, (_, A, B, _) in zip(init, chans)])

    def step(s, raw):
        outs, ns = [], []
        for i, (rate, A, B, C) in enumerate(chans):
            rl = _rl(s[i, 0], raw[i], rate, dt)
            m = A @ s[i, 1:3] + B * rl
            outs.append(C @ m)
            ns.append(jnp.concatenate([jnp.array([rl]), m]))
        return jnp.stack(ns), jnp.stack(outs)

    return state0, step


def make_simulator(controller: Callable, ctrl_state0: Any, cfg: ModelConfig,
                   dt: float = 0.05, decimate: int = 20, ctrl_every: int = 10,
                   solver: str = "ode4"):
    """Build a JIT-compiled simulate(params, x0, n_frames) -> telemetry dict.

    Telemetry arrays have leading axis n_frames (time = tel["t"]).
    solver: key of SOLVERS ("ode1", "ode2", "ode4").
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver '{solver}' unknown; use one of {list(SOLVERS)}")
    step = SOLVERS[solver]

    def inner_step(i, carry, p):
        x, cs, cmds, t = carry
        run_ctrl = (i % ctrl_every) == 0
        new_cmds, new_cs = controller(t, x, cs, p)
        cmds = jnp.where(run_ctrl, new_cmds, cmds)
        cs = jax.tree_util.tree_map(
            lambda a, b: jnp.where(run_ctrl, a, b), new_cs, cs)
        x = move_actuators(x, cmds, dt, p)
        x = step(x, dt, p, cfg)
        return (x, cs, cmds, t + dt)

    def frame(carry, _, p):
        carry = jax.lax.fori_loop(
            0, decimate, lambda i, c: inner_step(i, c, p), carry)
        x, cs, cmds, t = carry
        _, tel = dynamics(x, p, cfg)          # one diagnostic eval per frame
        tel = dict(tel)
        tel["t"] = t
        tel["cmds"] = cmds
        tel["ctrl_state"] = (jnp.ravel(cs) if hasattr(cs, "ravel")
                             else jnp.zeros(1))   # controller internals
        #  (integrators, reference-shaper states...) -- scope-able like
        #  any other signal: scope(tel, ["ctrl_state:0"])
        return carry, tel

    @functools.partial(jax.jit, static_argnums=(2,))
    def simulate(p: GliderParams, x0: jnp.ndarray, n_frames: int):
        cmds0, _ = controller(0.0, x0, ctrl_state0, p)
        carry0 = (x0, ctrl_state0, cmds0, 0.0)
        _, tel = jax.lax.scan(
            lambda c, _: frame(c, _, p), carry0, None, length=n_frames)
        return tel

    return simulate


def make_batch_simulator(simulate):
    """vmap over a batch of parameter sets AND initial states.

    Any leaf of GliderParams may carry a leading batch axis (use
    jax.tree_util.tree_map to broadcast), x0 has shape (B, 15).
    """
    return jax.jit(jax.vmap(simulate, in_axes=(0, 0, None)),
                   static_argnums=(2,))


def actuator_steady_state(p: GliderParams, rc, gamma, dB):
    """Actuator states x[12:23] already settled at constant commands
    [rc [m], gamma [rad], dB [N]] (same saturations as move_actuators), so a
    simulation can start with the actuators in place instead of at neutral."""
    def ss(A, B, u):                       # discrete steady state (I-A) x = B u
        return jnp.linalg.solve(jnp.eye(A.shape[0]) - A, B * u)
    m_rl = jnp.clip(rc, -p.dx_lin_max, p.dx_lin_max)
    g_pre = p.rotpre_b * gamma / (1.0 - p.rotpre_a)
    g_rl = p.rotpre_c * g_pre
    g_sat = jnp.clip(g_rl, -p.gamma_max, p.gamma_max)
    v_pre = p.vbdpre_b * dB / (1.0 - p.vbdpre_a)
    v_rl = p.vbdpre_c * v_pre
    return jnp.concatenate([
        jnp.array([m_rl]), ss(p.mass_A, p.mass_B, m_rl),
        jnp.array([g_pre, g_rl]), ss(p.rot_A, p.rot_B, g_sat),
        jnp.array([v_pre, v_rl]), ss(p.vbd_A, p.vbd_B, v_rl)])


def build_initial_state(p: GliderParams, pos_ned=(0.0, 0.0, 0.0),
                        euler=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
                        vel_frame="body", omega=(0.0, 0.0, 0.0),
                        actuators=None, ecef0=None):
    """Full initial state (26,).
    pos_ned [m] NED position; euler [rad] (roll, pitch, yaw);
    vel [m/s]: vel_frame="body" -> (u, v, w) relative to the water (the
        state), vel_frame="ned" -> (vN, vE, vD) over ground (the current
        p.current_n is subtracted);
    omega [rad/s] (p, q, r) body rates;
    actuators: None = neutral, or (rc [m], gamma [rad], dB [N]) already
        settled (see actuator_steady_state)."""
    from .dynamics import R_body_to_ned
    x0 = jnp.zeros(26)
    eul = jnp.asarray(euler, dtype=float)
    vel = jnp.asarray(vel, dtype=float)
    if vel_frame == "ned":
        Rnb = R_body_to_ned(eul[0], eul[1], eul[2]).T
        vel = Rnb @ (vel - p.current_n)
    elif vel_frame != "body":
        raise ValueError("vel_frame must be 'body' or 'ned'")
    x0 = (x0.at[0:3].set(jnp.asarray(pos_ned, dtype=float))
            .at[3:6].set(eul).at[6:9].set(vel)
            .at[9:12].set(jnp.asarray(omega, dtype=float)))
    if actuators is not None:
        x0 = x0.at[12:23].set(actuator_steady_state(p, *actuators))
    if ecef0 is not None:
        x0 = x0.at[23:26].set(jnp.asarray(ecef0))
    return x0


def initial_state(depth=0.0, pitch=0.0, u=0.0, roll=0.0, ecef0=None):
    x0 = jnp.zeros(26)
    if ecef0 is not None:
        x0 = x0.at[23:26].set(jnp.asarray(ecef0))
    x0 = (x0.at[2].set(depth).at[3].set(roll)
            .at[4].set(pitch).at[6].set(u))
    return x0
