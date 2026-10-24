"""Batched simulation with vmap.

(a) Family of open-loop descents over dx_lin  -> the report's Fig. 6.5 family
(b) Monte Carlo over hydrodynamic-coefficient uncertainty (+-10% on KL, KM, KD)
    -> distribution of glide path angle / speed; throughput benchmark.

On GPU the same code runs with batch sizes of 10^3-10^5.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
from gliderjax import (ModelConfig, seawing_like_defaults,
                       make_simulator, initial_state)

p0 = seawing_like_defaults()
cfg = ModelConfig()


def controller(t, x, cs, params):
    return jnp.array([cs[0], 0.0, -2.5]), cs   # dx command stored in ctrl state


DT, DEC = 0.05, 20
simulate = make_simulator(controller, ctrl_state0=jnp.array([0.03]), cfg=cfg,
                          dt=DT, decimate=DEC, ctrl_every=10)

# ---------------------------------------------------------------- (a) family
dx_family = jnp.array([0.010, 0.020, 0.030, 0.040, 0.050])
B = dx_family.shape[0]
params_b = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (B,) + jnp.shape(a)), p0)
x0 = initial_state(depth=0.0, pitch=-0.3, u=0.25).at[8].set(0.03)
x0_b = jnp.broadcast_to(x0, (B, 26))

from gliderjax.sim import make_simulator as _mk


def run_family(dx_vals, n_frames=300):
    # One compile per family member (5 here). For large batches, vmap over
    # a params field instead (see Monte Carlo section below).
    tels = []
    for dxv in dx_vals:
        def ctrl(t, x, cs, params, _dx=float(dxv)):
            return jnp.array([_dx, 0.0, -2.5]), cs
        sim = _mk(ctrl, jnp.zeros(1), cfg, dt=DT, decimate=DEC, ctrl_every=10)
        tels.append(sim(p0, x0, n_frames))
    return tels


tels = run_family(dx_family)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.8))
for dxv, tel in zip(np.asarray(dx_family), tels):
    e = np.asarray(tel["eta"])
    ax.plot(e[:, 0], e[:, 2], lw=1.2, label=f"dx={dxv*1000:.0f} mm")
ax.invert_yaxis(); ax.set_xlabel("X [m]"); ax.set_ylabel("depth [m]")
ax.grid(alpha=0.3); ax.legend(); ax.set_title("open-loop descent family (cf. report Fig. 6.5)")
outdir = os.path.join(os.path.dirname(__file__), "..", "out")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "batch_family_xz.png"), dpi=130)
print("wrote batch_family_xz.png")

# ------------------------------------------------------- (b) Monte Carlo vmap
BMC = 256
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
pert = lambda k: 1.0 + 0.10 * jax.random.normal(k, (BMC,))
params_mc = jax.tree_util.tree_map(
    lambda a: jnp.broadcast_to(a, (BMC,) + jnp.shape(a)), p0)
params_mc = params_mc._replace(
    KL=p0.KL * pert(k1), KM=p0.KM * pert(k2), KD=p0.KD * pert(k3))
x0_mc = jnp.broadcast_to(x0, (BMC, 26))

batch_sim = jax.jit(jax.vmap(simulate, in_axes=(0, 0, None)), static_argnums=(2,))
N_FR = 300                                    # 300 s each

t0 = time.time()
tel_mc = batch_sim(params_mc, x0_mc, N_FR)
tel_mc["t"].block_until_ready()
t_c = time.time() - t0
t0 = time.time()
tel_mc = batch_sim(params_mc, x0_mc, N_FR)
tel_mc["t"].block_until_ready()
t_r = time.time() - t0
steps = BMC * N_FR * DEC
print(f"Monte Carlo: {BMC} trajectories x {N_FR*DEC*DT:.0f} s "
      f"in {t_r:.2f} s  ({steps/t_r/1e6:.2f} M steps/s aggregate)")

pitch_ss = np.asarray(tel_mc["eta"])[:, -1, 4] * 57.3
V_ss = np.asarray(tel_mc["V"])[:, -1]
glide = np.asarray(tel_mc["eta"])[:, -1, 0] / np.asarray(tel_mc["eta"])[:, -1, 2]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax_, data, lbl in zip(axes, [pitch_ss, V_ss, glide],
                          ["steady pitch [deg]", "steady V [m/s]", "glide ratio X/Z"]):
    ax_.hist(data, bins=24, alpha=0.8)
    ax_.set_xlabel(lbl); ax_.grid(alpha=0.3)
fig.suptitle(f"Monte Carlo N={BMC}: +-10% on KL, KM, KD")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "batch_montecarlo.png"), dpi=130)
print("wrote batch_montecarlo.png")
print(f"glide ratio: mean {glide.mean():.2f}, std {glide.std():.2f}")
