"""Scopes: Simulink-style signal visualization for telemetry dicts.

Usage:
    tel = simulate(params, x0, n_frames)
    list_signals(tel)                       # what can I plot?
    scope(tel, ["nu_r:0", "alpha", "tau_flow_body:4"], title="pitch loop")
    default_scopes(tel, outdir="out")       # the standard debug figure set

Signal syntax: "name" for scalars, "name:i" for component i of a vector.
Derived signals: alpha_deg, beta_deg, roll_deg, pitch_deg, yaw_deg
(unwrapped), depth, path_deg (flight-path angle, + = climbing), depth_rate
(dz/dt, + = sinking), speed_h (horizontal speed).

Figures are closed after saving. Set KEEP_OPEN = True and call plt.show()
at the end of the script to inspect them (zoom/pan, like MATLAB figures).
Most time plots accept phases=[(t_start, name), ...] and mark each phase.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

RAD2DEG = 180.0 / np.pi
KEEP_OPEN = False


def _vel_n(t):
    return np.asarray(t["vel_n"])


_DERIVED = {
    "alpha_deg": lambda t: np.asarray(t["alpha"]) * RAD2DEG,
    "beta_deg": lambda t: np.asarray(t["beta"]) * RAD2DEG,
    "roll_deg": lambda t: np.asarray(t["eta"])[:, 3] * RAD2DEG,
    "pitch_deg": lambda t: np.asarray(t["eta"])[:, 4] * RAD2DEG,
    "yaw_deg": lambda t: np.unwrap(np.asarray(t["eta"])[:, 5]) * RAD2DEG,
    "depth": lambda t: np.asarray(t["eta"])[:, 2],
    "path_deg": lambda t: np.arctan2(-_vel_n(t)[:, 2],
                                     np.hypot(_vel_n(t)[:, 0], _vel_n(t)[:, 1])) * RAD2DEG,
    "depth_rate": lambda t: _vel_n(t)[:, 2],
    "speed_h": lambda t: np.hypot(_vel_n(t)[:, 0], _vel_n(t)[:, 1]),
}

LABELS6 = ["X [N]", "Y [N]", "Z [N]", "K [N m]", "M [N m]", "N [N m]"]
FLOW_LABELS6 = ["-D drag [N]", "SF side [N]", "-L lift [N]",
                "T1 roll [N m]", "T2 pitch [N m]", "T3 yaw [N m]"]
NU = ["u", "v", "w", "p", "q", "r"]

# axis labels with units for the common keys (fallback: the key itself)
LABELS = {
    "nu_r:0": "u [m/s]", "nu_r:1": "v [m/s]", "nu_r:2": "w [m/s]",
    "nu_r:3": "p [rad/s]", "nu_r:4": "q [rad/s]", "nu_r:5": "r [rad/s]",
    "eta:0": "x north [m]", "eta:1": "y east [m]", "eta:2": "z depth [m]",
    "depth": "depth [m]", "V": "V [m/s]",
    "alpha_deg": "alpha [deg]", "beta_deg": "beta [deg]",
    "roll_deg": "roll [deg]", "pitch_deg": "pitch [deg]", "yaw_deg": "yaw [deg]",
    "path_deg": "path angle [deg]", "depth_rate": "dz/dt [m/s]",
    "speed_h": "V horiz [m/s]",
    "actuators:0": "rc [m]", "actuators:1": "gamma [rad]",
    "actuators:2": "piston dxv [m]",
    "cmds:0": "rc cmd [m]", "cmds:1": "gamma cmd [rad]", "cmds:2": "dB cmd [N]",
    "B_net": "net buoyancy [N]",
}

# every term of the right-hand side of M nu_dot = sum(...) in dynamics()
WRENCH_TERMS = ["tau_flow_body", "tau_drot", "tau_surface", "tau_crossflow",
                "tau_coriolis_rb", "tau_CA", "tau_restoring"]


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


# ----------------------------------------------------------------------------
# saving / loading / external references
# ----------------------------------------------------------------------------
def save_tel(tel, path):
    """Save telemetry to .npz, to re-plot later without re-simulating."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in tel.items()})


def load_tel(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def from_simulink_csv(path):
    """Simulink log -> telemetry-like dict usable by scope()/compare().
    CSV columns: t, x, y, z, phi, theta, psi, u, v, w, p, q, r (the format
    of validation/compare_simulink.py). Flow angles are recomputed from
    u, v, w; vel_n by differentiating the position."""
    m = np.loadtxt(path, delimiter=",")
    t, eta, nu = m[:, 0], m[:, 1:7], m[:, 7:13]
    u, v, w = nu[:, 0], nu[:, 1], nu[:, 2]
    V = np.sqrt(u * u + v * v + w * w)
    return {"t": t, "eta": eta, "nu_r": nu, "V": V,
            "alpha": np.arctan2(w, u),
            "beta": np.arcsin(np.clip(v / np.maximum(V, 1e-9), -1, 1)),
            "vel_n": np.gradient(eta[:, 0:3], t, axis=0)}


# ----------------------------------------------------------------------------
# figure helpers
# ----------------------------------------------------------------------------
def _finish(fig, save, tight=True):
    if tight:
        fig.tight_layout()
    if save:
        d = os.path.dirname(save)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save, dpi=130)
        if not KEEP_OPEN:
            plt.close(fig)
    return fig


def _mark_phases(axes, phases, label_ax=None):
    """Dotted vertical line at each phase start; names on label_ax."""
    if not phases:
        return
    for ax in axes:
        for t0, _ in phases:
            if t0 > 0:
                ax.axvline(t0, color="0.5", ls=":", lw=0.9)
    if label_ax is not None:
        for t0, name in phases:
            label_ax.annotate(name, (t0, 1.0), xycoords=("data", "axes fraction"),
                              xytext=(3, 2), textcoords="offset points",
                              fontsize=7, color="0.35", va="bottom")


# ----------------------------------------------------------------------------
# generic scopes
# ----------------------------------------------------------------------------
def scope(tel, keys, title="", save=None, sharex=True, phases=None):
    """One stacked figure, one axis per key -- exactly a Simulink scope."""
    t = np.asarray(tel["t"])
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.9 * n),
                             sharex=sharex, squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        ax.plot(t, get_signal(tel, key), lw=1.0)
        ax.set_ylabel(LABELS.get(key, key), fontsize=8)
        ax.grid(alpha=0.3)
    _mark_phases(axes[:, 0], phases, axes[0, 0])
    axes[-1, 0].set_xlabel("t [s]")
    if title:
        fig.suptitle(title)
    return _finish(fig, save)


def compare(tels, keys, labels=None, title="", save=None, phases=None):
    """Overlay several runs, one axis per key (JAX vs Simulink, before vs
    after a parameter change...). Each run is a telemetry dict with its own
    "t", so different sample rates are fine."""
    labels = labels or [f"run {i}" for i in range(len(tels))]
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.9 * n), sharex=True,
                             squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        for tel, lab in zip(tels, labels):
            ax.plot(np.asarray(tel["t"]), get_signal(tel, key), lw=1.0, label=lab)
        ax.set_ylabel(LABELS.get(key, key), fontsize=8)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    _mark_phases(axes[:, 0], phases, axes[0, 0])
    axes[-1, 0].set_xlabel("t [s]")
    if title:
        fig.suptitle(title)
    return _finish(fig, save)


def scope_vector6(tel, name, title=None, save=None, labels=None, phases=None):
    """All six components of a wrench/velocity in one figure."""
    t = np.asarray(tel["t"])
    arr = np.asarray(tel[name])
    labels = labels or LABELS6
    fig, axes = plt.subplots(6, 1, figsize=(10, 11), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, arr[:, i], lw=1.0)
        ax.set_ylabel(labels[i], fontsize=8)
        ax.grid(alpha=0.3)
    _mark_phases(axes, phases, axes[0])
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(title or name)
    return _finish(fig, save)


def flow_frame_wrench(tel, save=None, phases=None):
    """Hydrodynamic wrench in the FLOW frame [-D, SF, -L, T1, T2, T3]: the
    Fh, Th vectors of the Simulink damping fcn, before rotating to body."""
    return scope_vector6(tel, "tau_flow_flow",
                         "hydrodynamic wrench, flow frame (Fh, Th)",
                         save=save, labels=FLOW_LABELS6, phases=phases)


def moments_breakdown(tel, axis=4, save=None, phases=None):
    """Overlay every force/moment contribution about one axis (default
    pitch=4) plus their sum (= M nu_dot, ~0 in steady state). Terms that are
    switched off (identically zero) are skipped.
    THE debugging plot: shows which physics dominates the balance."""
    t = np.asarray(tel["t"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    total = np.zeros(t.shape)
    for name in WRENCH_TERMS:
        if name not in tel:
            continue
        y = np.asarray(tel[name])[:, axis]
        total = total + y
        if np.any(y != 0.0):
            ax.plot(t, y, lw=1.1, label=name)
    ax.plot(t, total, "k--", lw=0.9, label="sum (= M nu_dot)")
    _mark_phases([ax], phases, ax)
    ax.set_xlabel("t [s]"); ax.set_ylabel(LABELS6[axis])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"balance, axis {LABELS6[axis]}", pad=14 if phases else None)
    return _finish(fig, save)


# ----------------------------------------------------------------------------
# glider-specific views
# ----------------------------------------------------------------------------
def overview(tel, title="", save=None, phases=None):
    """One-page summary: depth, pitch/alpha/path angle, roll, yaw vs time |
    top view, speed, vertical speed."""
    t = np.asarray(tel["t"])
    e = np.asarray(tel["eta"])
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.6, 1])
    left = [fig.add_subplot(gs[0, 0])]
    left += [fig.add_subplot(gs[i, 0], sharex=left[0]) for i in range(1, 4)]
    left[0].plot(t, e[:, 2], lw=1.0)
    left[0].invert_yaxis(); left[0].set_ylabel("depth [m]")
    for key in ["pitch_deg", "alpha_deg", "path_deg"]:
        left[1].plot(t, get_signal(tel, key), lw=1.0, label=key[:-4])
    left[1].set_ylabel("[deg]"); left[1].legend(fontsize=7, ncol=3)
    left[2].plot(t, get_signal(tel, "roll_deg"), lw=1.0)
    left[2].set_ylabel("roll [deg]")
    left[3].plot(t, get_signal(tel, "yaw_deg"), lw=1.0)
    left[3].set_ylabel("yaw [deg]"); left[3].set_xlabel("t [s]")
    xy = fig.add_subplot(gs[0:2, 1])
    xy.plot(e[:, 1], e[:, 0], lw=1.0)
    xy.plot(e[0, 1], e[0, 0], "go", ms=5, label="start")
    xy.plot(e[-1, 1], e[-1, 0], "rs", ms=5, label="end")
    xy.set_xlabel("East [m]"); xy.set_ylabel("North [m]")
    xy.axis("equal"); xy.legend(fontsize=7); xy.set_title("top view", fontsize=9)
    sp = fig.add_subplot(gs[2, 1], sharex=left[0])
    sp.plot(t, np.asarray(tel["V"]), lw=1.0); sp.set_ylabel("V [m/s]")
    vz = fig.add_subplot(gs[3, 1], sharex=left[0])
    vz.plot(t, get_signal(tel, "depth_rate"), lw=1.0)
    vz.set_ylabel("dz/dt [m/s] (+ down)"); vz.set_xlabel("t [s]")
    for ax in fig.axes:
        ax.grid(alpha=0.3)
    _mark_phases(left + [sp, vz], phases, left[0])
    if title:
        fig.suptitle(title)
    return _finish(fig, save)


def actuators_plot(tel, title="actuators: position vs command", save=None,
                   phases=None):
    """Each actuator against its command in the same units: moving mass rc
    [mm], rotary mass gamma [deg], VBD as net buoyancy vs dB command [N]
    (they differ when the piston saturates), and the piston itself [mm]."""
    t = np.asarray(tel["t"])
    a, c = np.asarray(tel["actuators"]), np.asarray(tel["cmds"])
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)
    axes[0].plot(t, a[:, 0] * 1e3, lw=1.0, label="rc")
    axes[0].plot(t, c[:, 0] * 1e3, "--", lw=1.0, label="command")
    axes[0].set_ylabel("rc [mm]")
    axes[1].plot(t, a[:, 1] * RAD2DEG, lw=1.0, label="gamma")
    axes[1].plot(t, c[:, 1] * RAD2DEG, "--", lw=1.0, label="command")
    axes[1].set_ylabel("gamma [deg]")
    if "B_net" in tel:
        axes[2].plot(t, np.asarray(tel["B_net"]), lw=1.0, label="net buoyancy")
    axes[2].plot(t, c[:, 2], "--", lw=1.0, label="dB command")
    axes[2].set_ylabel("[N]")
    axes[3].plot(t, a[:, 2] * 1e3, lw=1.0)
    axes[3].set_ylabel("piston dxv [mm]"); axes[3].set_xlabel("t [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
    for ax in axes[:3]:
        ax.legend(fontsize=7, loc="best")
    _mark_phases(axes, phases, axes[0])
    fig.suptitle(title)
    return _finish(fig, save)


def trajectory_xz(tel, save=None):
    e = np.asarray(tel["eta"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(e[:, 0], e[:, 2], lw=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("X north [m]"); ax.set_ylabel("depth [m]")
    ax.grid(alpha=0.3)
    return _finish(fig, save)


def trajectory_xy(tel, save=None):
    """Top view: East right, North up, equal scale."""
    e = np.asarray(tel["eta"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(e[:, 1], e[:, 0], lw=1.2)
    ax.plot(e[0, 1], e[0, 0], "go", ms=6, label="start")
    ax.plot(e[-1, 1], e[-1, 0], "rs", ms=6, label="end")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return _finish(fig, save)


def trajectory_3d(tel, save=None):
    e = np.asarray(tel["eta"])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot(e[:, 1], e[:, 0], e[:, 2], lw=1.0)
    ax.scatter(e[0, 1], e[0, 0], e[0, 2], color="g", label="start")
    ax.scatter(e[-1, 1], e[-1, 0], e[-1, 2], color="r", marker="s", label="end")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("depth [m]")
    ax.invert_zaxis(); ax.legend(fontsize=8)
    return _finish(fig, save, tight=False)


# ----------------------------------------------------------------------------
# numbers
# ----------------------------------------------------------------------------
def phase_summary(tel, phases, t_end=None, frac=0.3):
    """Quasi-steady values per phase: mean over the last `frac` of each
    phase. phases = [(t_start, name), ...]. Prints a table and returns a
    list of dicts. path = flight-path angle (+ climbing); glide = horizontal
    / vertical speed; yaw_rate from the unwrapped yaw."""
    t = np.asarray(tel["t"])
    t_end = float(t[-1]) if t_end is None else t_end
    yaw = np.unwrap(np.asarray(tel["eta"])[:, 5])
    sig = {
        "pitch[deg]": get_signal(tel, "pitch_deg"),
        "alpha[deg]": get_signal(tel, "alpha_deg"),
        "path[deg]": get_signal(tel, "path_deg"),
        "V[m/s]": np.asarray(tel["V"]),
        "dz/dt[m/s]": get_signal(tel, "depth_rate"),
        "roll[deg]": get_signal(tel, "roll_deg"),
        "yaw_rate[deg/s]": np.gradient(yaw, t) * RAD2DEG,
    }
    v_h = get_signal(tel, "speed_h")
    ends = [ph[0] for ph in phases[1:]] + [t_end]
    rows = []
    for (t0, name), t1 in zip(phases, ends):
        w0 = t1 - frac * (t1 - t0)
        m = (t >= w0) & (t <= t1)
        if not m.any():
            continue
        row = {"phase": name, "window": (w0, t1)}
        row.update({k: float(np.mean(v[m])) for k, v in sig.items()})
        row["glide"] = float(np.mean(v_h[m])) / max(abs(row["dz/dt[m/s]"]), 1e-9)
        rows.append(row)
    cols = list(sig) + ["glide"]
    wn = max([5] + [len(r["phase"]) for r in rows])
    print(f"quasi-steady values (mean over the last {frac:.0%} of each phase)")
    print(f"{'phase':<{wn}}  {'window [s]':>13}  "
          + "  ".join(f"{c:>{max(len(c), 8)}}" for c in cols))
    for r in rows:
        win = f"{r['window'][0]:.0f}-{r['window'][1]:.0f}"
        print(f"{r['phase']:<{wn}}  {win:>13}  "
              + "  ".join(f"{r[c]:>+{max(len(c), 8)}.3f}" for c in cols))
    return rows


def default_scopes(tel, outdir="out", prefix="", phases=None):
    os.makedirs(outdir, exist_ok=True)
    p = lambda s: os.path.join(outdir, prefix + s)
    overview(tel, title="overview", save=p("overview.png"), phases=phases)
    scope(tel, [f"nu_r:{i}" for i in range(6)],
          title="body-frame relative velocities " + str(NU),
          save=p("scope_body_vel.png"), phases=phases)
    scope(tel, [f"nu_r_dot:{i}" for i in range(6)],
          title="body-frame accelerations", save=p("scope_body_acc.png"),
          phases=phases)
    scope(tel, ["pos_n:0", "pos_n:1", "depth", "vel_n:0", "vel_n:2",
                "accel_n:0", "accel_n:2"],
          title="inertial position / velocity / acceleration",
          save=p("scope_inertial.png"), phases=phases)
    scope(tel, ["roll_deg", "pitch_deg", "yaw_deg",
                "alpha_deg", "beta_deg", "V"],
          title="attitude & flow angles", save=p("scope_attitude.png"),
          phases=phases)
    actuators_plot(tel, save=p("scope_actuators.png"), phases=phases)
    flow_frame_wrench(tel, save=p("scope_tau_flow_frame.png"), phases=phases)
    scope_vector6(tel, "tau_flow_body", "hydrodynamic wrench (body frame)",
                  save=p("scope_tau_flow.png"), phases=phases)
    scope_vector6(tel, "tau_restoring", "restoring wrench (weight+buoyancy)",
                  save=p("scope_tau_restoring.png"), phases=phases)
    moments_breakdown(tel, axis=3, save=p("scope_roll_balance.png"), phases=phases)
    moments_breakdown(tel, axis=4, save=p("scope_pitch_balance.png"), phases=phases)
    moments_breakdown(tel, axis=5, save=p("scope_yaw_balance.png"), phases=phases)
    trajectory_xz(tel, save=p("trajectory_xz.png"))
    trajectory_xy(tel, save=p("trajectory_xy.png"))
    print(f"scopes written to {outdir}/")
