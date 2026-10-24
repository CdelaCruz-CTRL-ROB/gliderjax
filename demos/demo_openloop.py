"""Open-loop descent: dB = -2.5 N, dx_lin = +0.03 m, gamma = 0.

Mirrors the report's open-loop pitch/descent validation. Also prints wall
time for the compiled simulation (the no-GPU speedup demonstration).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
from gliderjax import (ModelConfig, seawing_like_defaults,
                       make_simulator, initial_state, scope)

p = seawing_like_defaults()
cfg = ModelConfig(use_parity_hydro=True, use_ca_omega=False)


def controller(t, x, cs, params):
    """Open loop: constant commands. cs unused (kept for API symmetry)."""
    cmds = jnp.array([0.03, 0.0, -2.5])
    return cmds, cs


DT, DEC = 0.05, 20                     # 20 steps/frame -> 1 Hz telemetry
simulate = make_simulator(controller, ctrl_state0=jnp.zeros(1), cfg=cfg,
                          dt=DT, decimate=DEC, ctrl_every=10)

x0 = initial_state(depth=0.0, pitch=-0.35, u=0.25)
x0 = x0.at[8].set(0.03)  # small +w: near glide equilibrium
T_SIM = 400.0
n_frames = int(T_SIM / (DT * DEC))

t0 = time.time()
tel = simulate(p, x0, n_frames)
tel["t"].block_until_ready()
t_compile = time.time() - t0

t0 = time.time()
tel = simulate(p, x0, n_frames)
tel["t"].block_until_ready()
t_run = time.time() - t0

steps = n_frames * DEC
print(f"simulated {T_SIM:.0f} s ({steps} RK4 steps)")
print(f"  first call (incl. JIT compile): {t_compile:.2f} s")
print(f"  compiled run:                   {t_run*1000:.1f} ms "
      f"({steps/t_run/1e6:.2f} M steps/s)")
print(f"  final: depth={float(tel['eta'][-1,2]):.1f} m, "
      f"pitch={float(tel['eta'][-1,4])*57.3:.1f} deg, "
      f"alpha={float(tel['alpha'][-1])*57.3:.2f} deg, "
      f"V={float(tel['V'][-1]):.3f} m/s, "
      f"X={float(tel['eta'][-1,0]):.1f} m")

outdir = os.path.join(os.path.dirname(__file__), "..", "out")
scope.default_scopes(tel, outdir=outdir, prefix="openloop_")
