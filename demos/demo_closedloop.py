"""Closed-loop pitch control: the report's PID-form backstepping controller
(eq. final_pid_structure) running as a DISCRETE controller, commanding the
rate-limited moving mass, through a dive/climb mission profile.

Controller state (pytree): [integral_of_error]  with conditional
anti-windup (integration frozen when the actuator command saturates).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
from gliderjax import (ModelConfig, seawing_like_defaults,
                       make_simulator, initial_state, scope,
                       make_ref_pipeline)
from gliderjax.params import G0

p = seawing_like_defaults()
cfg = ModelConfig()

# PID gains in the report's normalized form (units: m of mass travel / rad)
KP, KD_, KI = 0.015, 0.08, 0.003
CTRL_DT = 0.5                       # controller period [s]

# Reference shaping (Simulink values: pitch RL 0.1 rad/s -> 1/(5s+1)).
# ctrl_state = [integ, ref_pipeline_state(3,3) flattened]
REF0, REF_STEP = make_ref_pipeline(CTRL_DT)


def theta_ref(t):
    """Mission profile: dive at -30 deg, surface maneuver, climb at +30 deg."""
    th_dive, th_climb = -0.44, 0.44               # +-25 deg (feasible)
    return jnp.where(t < 220.0, th_dive,
           jnp.where(t < 240.0, 0.0, th_climb))


def dB_ref(t):
    return jnp.where(t < 220.0, -3.0,
           jnp.where(t < 240.0, 0.0, 3.0))


def controller(t, x, cs, params):
    th, q = x[4], x[10]
    integ, ref_s = cs[0], cs[1:10].reshape(3, 3)
    ref_s, shaped = REF_STEP(ref_s, jnp.array([theta_ref(t), 0.0, 0.0]))
    e = th - shaped[0]
    m_tot = params.m_static + params.m_lin + params.m_rot + params.m_pist
    zg0 = (params.m_static * params.r_static[2] + params.m_lin * params.r_lin0[2]
           + params.m_rot * params.R_rot) / m_tot     # gamma=0, piston z=0
    # x_g command (report eq. final_pid_structure, feed-forward from params)
    xg_cmd = (1.0 / jnp.cos(th)) * (
        KP * e + KD_ * q + KI * integ - zg0 * jnp.sin(th))
    dx_cmd = m_tot * xg_cmd / params.m_lin       # mass fwd (dx>0) => nose down
    sat = jnp.abs(dx_cmd) >= params.dx_lin_max
    integ = jnp.where(sat, integ, integ + e * CTRL_DT)   # anti-windup
    cmds = jnp.array([dx_cmd, 0.0, dB_ref(t)])
    return cmds, jnp.concatenate([jnp.array([integ]), ref_s.ravel()])


DT, DEC = 0.05, 20
simulate = make_simulator(controller, ctrl_state0=jnp.zeros(10), cfg=cfg,
                          dt=DT, decimate=DEC, ctrl_every=int(CTRL_DT / DT))

x0 = initial_state(depth=0.0, pitch=0.0, u=0.15)
T_SIM = 420.0
n = int(T_SIM / (DT * DEC))

tel = simulate(p, x0, n)
tel["t"].block_until_ready()
t0 = time.time(); tel = simulate(p, x0, n); tel["t"].block_until_ready()
print(f"closed-loop {T_SIM:.0f} s in {(time.time()-t0)*1000:.0f} ms")

import numpy as np
th = np.asarray(tel["eta"])[:, 4]
tt = np.asarray(tel["t"])
thr = np.asarray(theta_ref(jnp.asarray(tt)))   # RAW ref (shaped one is what
#                                                  the controller tracks)
rms = np.sqrt(np.mean((th[tt > 60] - thr[tt > 60]) ** 2)) * 57.3
print(f"pitch error vs RAW ref RMS (t>60s): {rms:.2f} deg "
      f"(includes reference-shaping lag -- by design)")
print(f"final depth {float(tel['eta'][-1,2]):.1f} m")

outdir = os.path.join(os.path.dirname(__file__), "..", "out")
scope.default_scopes(tel, outdir=outdir, prefix="closedloop_")

# tracking scope
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(tt, th * 57.3, label="pitch")
ax[0].plot(tt, thr * 57.3, "--", label="ref")
ax[0].set_ylabel("theta [deg]"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(tt, np.asarray(tel["actuators"])[:, 0] * 1000, label="dx_lin [mm]")
ax[1].plot(tt, np.asarray(tel["cmds"])[:, 0] * 1000, "--", lw=0.8, label="cmd")
ax[1].set_ylabel("moving mass [mm]"); ax[1].set_xlabel("t [s]")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "closedloop_tracking.png"), dpi=130)
print("wrote closedloop_tracking.png")
