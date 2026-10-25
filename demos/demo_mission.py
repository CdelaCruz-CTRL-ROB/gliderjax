"""Waypoint mission demo: the report's Veracruz triangle, flown end to end.

Author's FSM (gliderjax/fsm/simulink.py) -> reference shaping -> placeholder attitude cascade ->
actuator chains -> 6-DOF plant with surface physics and geodetic truth.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp
from gliderjax import ModelConfig, make_simulator, initial_state, scope, fsm
from gliderjax.params import cidesi_defaults
from gliderjax.autopilot import make_waypoint_controller, waypoint_signals
from gliderjax.navigation import MISSION_TRIANGLE, MISSION_SQUARE7, MISSION_SQUARE_3KM

DT, DEC, CTRL_EVERY = 0.05, 20, 5          # 0.5 s FSM/controller tick
CTRL_DT = DT * CTRL_EVERY

p = cidesi_defaults()
cfg = ModelConfig()

nav = fsm.make("simulink", MISSION_SQUARE_3KM)
ACCEPT_R = fsm.simulink.Config().acceptance_radius_m
cs0, controller = make_waypoint_controller(nav, CTRL_DT)
sim = make_simulator(controller, ctrl_state0=cs0, cfg=cfg,
                     dt=DT, decimate=DEC, ctrl_every=CTRL_EVERY)

x0 = initial_state(depth=0.0, ecef0=nav.ecef0)
T_SIM =80000.0
n = int(T_SIM / (DT * DEC))

t0 = time.time()
tel = sim(p, x0, n)
tel["t"].block_until_ready()
print(f"mission sim: {T_SIM:.0f} s in {time.time()-t0:.1f} s (incl. compile)")

# ---- mission analysis -------------------------------------------------------
tt = np.asarray(tel["t"])
eta = np.asarray(tel["eta"])
sig = waypoint_signals(tel, nav)          # FSM signals by name
fsm_state = sig["glider_state"]
wp_idx = sig["current_wp_index"]
Dn = np.column_stack([sig["D_norm_n"], sig["D_norm_e"]])
complete = sig["mission_complete"]
wps = nav.waypoints_ned

done_i = np.argmax(complete > 0.5) if complete.max() > 0.5 else -1
print(f"waypoints reached: {int(wp_idx.max()) - 1} / {len(MISSION_SQUARE_3KM)-0}"
      f" | mission complete: {'YES at t=%.0f s' % tt[done_i] if done_i > 0 else 'no'}")
print(f"max depth {eta[:,2].max():.1f} m | D_norm final "
      f"[{Dn[-1,0]:+.4f}, {Dn[-1,1]:+.4f}]")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(out, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.5, 7))
ax.plot(eta[:, 1], eta[:, 0], lw=1.0, label="track (flat NED)")
for i, w in enumerate(wps):
    ax.plot(w[1], w[0], "r*", ms=12)
    ax.add_patch(plt.Circle((w[1], w[0]), ACCEPT_R, fill=False,
                            color="r", ls="--", lw=0.7))
    ax.annotate(f"WP{i}", (w[1], w[0]), textcoords="offset points",
                xytext=(8, 8), fontsize=9)
ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
ax.set_title("waypoint mission (author's FSM)")
fig.tight_layout(); fig.savefig(os.path.join(out, "mission_track.png"), dpi=130)

fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axs[0].plot(tt, eta[:, 2]); axs[0].invert_yaxis()
axs[0].set_ylabel("depth [m]"); axs[0].grid(alpha=0.3)
axs[1].step(tt, fsm_state, where="post")
axs[1].set_ylabel("FSM state"); axs[1].set_yticks(range(1, 8))
axs[1].set_yticklabels(nav.state_names, fontsize=7); axs[1].grid(alpha=0.3)
axs[2].plot(tt, np.rad2deg(eta[:, 5]), label="yaw")
axs[2].plot(tt, np.rad2deg(eta[:, 3]), label="roll", lw=0.8)
axs[2].set_ylabel("[deg]"); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)
axs[3].plot(tt, Dn[:, 0], label="D_norm N")
axs[3].plot(tt, Dn[:, 1], label="D_norm E")
axs[3].set_ylabel("drift est."); axs[3].set_xlabel("t [s]")
axs[3].legend(fontsize=8); axs[3].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(out, "mission_profile.png"), dpi=130)
print("wrote mission_track.png, mission_profile.png")
