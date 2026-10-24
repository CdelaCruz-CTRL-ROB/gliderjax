"""Scopes: Simulink-style signal visualization for telemetry dicts.

Usage:
    tel = simulate(params, x0, n_frames)
    list_signals(tel)                       # what can I plot?
    scope(tel, ["nu_r:0", "alpha", "tau_flow_body:4"], title="pitch loop")
    default_scopes(tel, outdir="out")       # the standard debug figure set

Signal syntax: "name" for scalars, "name:i" for component i of a vector.
Derived signals available: alpha_deg, beta_deg, roll_deg, pitch_deg, yaw_deg.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAD2DEG = 180.0 / np.pi

_DERIVED = {
    "alpha_deg": lambda t: np.asarray(t["alpha"]) * RAD2DEG,
    "beta_deg": lambda t: np.asarray(t["beta"]) * RAD2DEG,
    "roll_deg": lambda t: np.asarray(t["eta"])[:, 3] * RAD2DEG,
    "pitch_deg": lambda t: np.asarray(t["eta"])[:, 4] * RAD2DEG,
    "yaw_deg": lambda t: np.asarray(t["eta"])[:, 5] * RAD2DEG,
    "depth": lambda t: np.asarray(t["eta"])[:, 2],
}

LABELS6 = ["X [N]", "Y [N]", "Z [N]", "K [N m]", "M [N m]", "N [N m]"]
NU = ["u", "v", "w", "p", "q", "r"]


def get_signal(tel, key):
    if key in _DERIVED:
        return _DERIVED[key](tel)
    name, _, idx = key.partition(":")
    arr = np.asarray(tel[name])
    return arr[:, int(idx)] if idx else arr


def list_signals(tel):
    out = []
    for k, v in tel.items():
        v = np.asarray(v)
        out.append(f"{k}  shape={v.shape[1:] if v.ndim > 1 else 'scalar'}")
    out += [f"[derived] {k}" for k in _DERIVED]
    print("\n".join(out))
    return out


def scope(tel, keys, title="", save=None, sharex=True):
    """One stacked figure, one axis per key -- exactly a Simulink scope."""
    t = np.asarray(tel["t"])
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.9 * n),
                             sharex=sharex, squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        ax.plot(t, get_signal(tel, key), lw=1.0)
        ax.set_ylabel(key, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("t [s]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        plt.close(fig)
    return fig


def scope_vector6(tel, name, title=None, save=None, labels=None):
    """All six components of a wrench/velocity in one figure."""
    t = np.asarray(tel["t"])
    arr = np.asarray(tel[name])
    labels = labels or LABELS6
    fig, axes = plt.subplots(6, 1, figsize=(10, 11), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, arr[:, i], lw=1.0)
        ax.set_ylabel(labels[i], fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(title or name)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        plt.close(fig)
    return fig


def moments_breakdown(tel, axis=4, save=None):
    """Overlay every moment contribution about one axis (default pitch=4).
    THE debugging plot: shows which physics dominates the balance."""
    t = np.asarray(tel["t"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in ["tau_flow_body", "tau_drot", "tau_coriolis_rb",
                 "tau_CA", "tau_restoring"]:
        ax.plot(t, np.asarray(tel[name])[:, axis], lw=1.1, label=name)
    ax.set_xlabel("t [s]"); ax.set_ylabel(LABELS6[axis])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"moment balance, axis {LABELS6[axis]}")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def trajectory_xz(tel, save=None):
    e = np.asarray(tel["eta"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(e[:, 0], e[:, 2], lw=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("X north [m]"); ax.set_ylabel("depth [m]")
    ax.grid(alpha=0.3); fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


def default_scopes(tel, outdir="out", prefix=""):
    os.makedirs(outdir, exist_ok=True)
    p = lambda s: os.path.join(outdir, prefix + s)
    scope(tel, [f"nu_r:{i}" for i in range(6)],
          title="body-frame relative velocities " + str(NU),
          save=p("scope_body_vel.png"))
    scope(tel, [f"nu_r_dot:{i}" for i in range(6)],
          title="body-frame accelerations", save=p("scope_body_acc.png"))
    scope(tel, ["pos_n:0", "pos_n:1", "depth", "vel_n:0", "vel_n:2",
                "accel_n:0", "accel_n:2"],
          title="inertial position / velocity / acceleration",
          save=p("scope_inertial.png"))
    scope(tel, ["roll_deg", "pitch_deg", "yaw_deg",
                "alpha_deg", "beta_deg", "V"],
          title="attitude & flow angles", save=p("scope_attitude.png"))
    scope(tel, ["actuators:0", "actuators:1", "actuators:2",
                "cmds:0", "cmds:1", "cmds:2"],
          title="actuators (positions & commands)", save=p("scope_actuators.png"))
    scope_vector6(tel, "tau_flow_body", "hydrodynamic wrench (body frame)",
                  save=p("scope_tau_flow.png"))
    scope_vector6(tel, "tau_restoring", "restoring wrench (weight+buoyancy)",
                  save=p("scope_tau_restoring.png"))
    moments_breakdown(tel, axis=4, save=p("scope_pitch_balance.png"))
    moments_breakdown(tel, axis=5, save=p("scope_yaw_balance.png"))
    trajectory_xz(tel, save=p("trajectory_xz.png"))
    print(f"scopes written to {outdir}/")
