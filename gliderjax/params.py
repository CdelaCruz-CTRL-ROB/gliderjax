"""Glider parameters (JAX pytree) and static model configuration.

All numeric values live in config/cidesi.toml (transcribed from the user's
Simulink blocks). This module only defines the structure and loads it:

    p, cfg = load_config("config/cidesi.toml", model="simulink", dt=0.05)

rho=1000 (freshwater) per the VBD block -- confirm consistency with CFD
density.
"""
from __future__ import annotations

from typing import NamedTuple
import dataclasses
import math
import os
import tomllib
import jax.numpy as jnp

G0 = 9.81
RHO = 1000.0          # per VBD block (freshwater)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "cidesi.toml")


class GliderParams(NamedTuple):
    # ----- four-mass budget (body frame, origin CO) ------------------------
    m_static: jnp.ndarray
    r_static: jnp.ndarray
    I_static: jnp.ndarray
    m_lin: jnp.ndarray
    r_lin0: jnp.ndarray        # neutral position [0.2861, 0, 0.001]
    I_lin: jnp.ndarray
    dx_lin_max: jnp.ndarray    # command saturation on r_c
    m_rot: jnp.ndarray
    x_rot: jnp.ndarray
    R_rot: jnp.ndarray
    I_rot: jnp.ndarray
    gamma_max: jnp.ndarray
    w_rot_max: jnp.ndarray
    m_pist: jnp.ndarray        # VBD piston moving mass
    I_pist: jnp.ndarray

    # ----- VBD geometry (exact port of fcn_VBD_actuator) -------------------
    vbd_area: jnp.ndarray      # piston area
    vbd_max_disp: jnp.ndarray  # half stroke
    vbd_x0: jnp.ndarray        # piston station
    vbd_rate: jnp.ndarray      # piston speed limit [m/s] (slew on delta_x)

    # ----- ACTUATOR CHAINS (ZOH-exact discrete) -----------------------------
    # Every channel:  [prefilter TF] -> [rate limiter] -> [main TF] -> output
    #
    # TFs are given to cidesi_defaults() as CONTINUOUS num/den coefficient
    # lists, exactly as written in the Simulink Transfer Fcn blocks:
    #     400/(s^2+30s+400)  ->  ([400.0], [1.0, 30.0, 400.0])
    #     1/(2s+1)           ->  ([1.0],   [2.0, 1.0])
    #     4/(s+4)            ->  ([4.0],   [1.0, 4.0])
    # Internally: tf2ss + matrix-exponential ZOH -> exact discrete (A,B,C).
    # First-order mains are padded with a pole at 1e4 rad/s to keep a fixed
    # 2-state size (effect < 1e-200 at dt=0.05). Rate limiter:
    #     y+ = y + clip(u - y, +-R*dt);  R huge => stage OFF.
    mass_rate: jnp.ndarray     # [m/s] moving-mass rate limit (0.008 physical)
    mass_A: jnp.ndarray        # (2,2) main TF states
    mass_B: jnp.ndarray        # (2,)
    mass_C: jnp.ndarray        # (2,)  output = C @ x
    rotpre_a: jnp.ndarray      # rotary prefilter 1-state ZOH (scalars a,b,c)
    rotpre_b: jnp.ndarray
    rotpre_c: jnp.ndarray
    rot_rate: jnp.ndarray      # [rad/s]                          # <-- CONFIRM
    rot_A: jnp.ndarray         # (2,2) rotary main TF
    rot_B: jnp.ndarray
    rot_C: jnp.ndarray
    vbdpre_a: jnp.ndarray      # VBD prefilter (Simulink helper 1/(0.2s+1))
    vbdpre_b: jnp.ndarray
    vbdpre_c: jnp.ndarray
    vbd_rate_N: jnp.ndarray    # [N/s] rate limit on dB signal    # <-- CONFIRM
    vbd_A: jnp.ndarray         # (2,2) VBD main TF (1/(2s+1) padded)
    vbd_B: jnp.ndarray
    vbd_C: jnp.ndarray
    # ----- added mass (Simulink values, port-starboard symmetry imposed) ----
    MA: jnp.ndarray            # (6,6)
    MA_hull: jnp.ndarray

    # ----- flow-frame coefficients (user's damping fcn) ---------------------
    KD0: jnp.ndarray
    KD: jnp.ndarray
    KL0: jnp.ndarray
    KL: jnp.ndarray
    KM0: jnp.ndarray
    KM: jnp.ndarray
    Kbeta: jnp.ndarray         # side force
    KMR_mag: jnp.ndarray       # roll moment magnitude (signum mode) / K_e
    KMR_o: jnp.ndarray         # parity-odd part (research mode)
    KMY_lin: jnp.ndarray       # yaw stiffness linear
    KMY_cub: jnp.ndarray       # yaw stiffness cubic (sign-reversal ~11 deg)

    # ----- rotational damping (user's D_body: speed-independent) -----------
    Kp_lin: jnp.ndarray
    Kq_lin: jnp.ndarray
    Kr_lin: jnp.ndarray
    Kp_quad: jnp.ndarray
    Kq_quad: jnp.ndarray
    Kr_quad: jnp.ndarray
    KpV: jnp.ndarray           # ZERO for Simulink validation
    KqV: jnp.ndarray
    KrV: jnp.ndarray

    # ----- surface transit physics (identify via pool bobbing test) --------
    hull_L: jnp.ndarray        # [m] hull length (surface scaler geometry)
    hull_H: jnp.ndarray        # [m] hull height
    B33_rad: jnp.ndarray       # [N s/m] heave radiation damping at surface
    B55_rad: jnp.ndarray       # [N m s] pitch radiation damping at surface
    dmA_dz: jnp.ndarray        # [kg/m] water-entry added-mass gradient
    ma_scale_floor: jnp.ndarray  # min added-mass scale when emerged
    # ----- crossflow drag (large-incidence blend; retires envelope clamp) --
    Cd_cross_y: jnp.ndarray    # [kg/m] quadratic sway drag  ~0.5*rho*Cd*A_side
    Cd_cross_z: jnp.ndarray    # [kg/m] quadratic heave drag ~0.5*rho*Cd*A_plan
    chi_blend: jnp.ndarray     # [rad] incidence where crossflow blends in
    chi_width: jnp.ndarray     # [rad] blend width
    # ----- GPS sensor --------------------------------------------------------
    gps_sigma_m: jnp.ndarray   # horizontal fix noise 1-sigma [m] (0 = ideal)
    gps_gate_depth: jnp.ndarray  # fixes available only when z < this [m]
    # ----- disturbances & environment ---------------------------------------
    disturb_xg: jnp.ndarray    # CG disturbance [m] (restoring + coriolis
    disturb_yg: jnp.ndarray    #  only, replicating the Simulink wiring)
    disturb_xlin: jnp.ndarray  # moving-mass position disturbance [m]: added
    #                            to r_lin, so it enters M, I_b and r_g
    current_n: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    kmr_mode: str = "signum"         # "signum" (Simulink) | "parity" | "const"
    kmy_cubic: bool = True
    use_alpha_saturation: bool = False  # tanh clamp of alpha/beta at ~25 deg
    #                                     inside tau_flowframe. NOT in Simulink
    #                                     (added by the port as a guard).
    use_ca_omega: bool = False
    use_ca_translational: bool = False  # Munk block: double counts vs CFD
    use_rot_damping: bool = True
    surface_model: str = "full"         # "none" | "simulink" (two-point
    #                                     fcn_buoyancy_surface) | "full" (9
    #                                     stations + added mass + entry/radiation)
    use_crossflow: bool = True          # large-incidence quadratic drag blend
    use_geodesy: bool = True            # integrate ECEF truth position
    replicate_disturb_wiring: bool = True  # disturb_xg/yg enter C_RB +
    #                                        restoring, NOT M_RB (Simulink)


def zoh_tf(num, den, dt, n_states):
    """Exact ZOH discretization of a strictly proper continuous TF num/den.
    Pads with poles at 1e4 rad/s up to n_states so JAX shapes stay fixed.
    Returns (Ad (n,n), Bd (n,), Cd (n,)); discrete output y_k = Cd @ x_k."""
    import numpy as np
    from scipy.linalg import expm
    from scipy.signal import tf2ss
    num = np.atleast_1d(np.asarray(num, dtype=float))
    den = np.atleast_1d(np.asarray(den, dtype=float))
    assert len(num) < len(den), "TF must be strictly proper (deg num < deg den)"
    while len(den) - 1 < n_states:
        den = np.polymul(den, [1e-4, 1.0])     # (s/1e4 + 1): fast, unity DC
    A, B, C, D = tf2ss(num, den)
    assert abs(float(np.squeeze(D))) < 1e-12
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1)); M[:n, :n] = A; M[:n, n:] = B
    Md = expm(M * dt)
    return (jnp.asarray(Md[:n, :n]), jnp.asarray(Md[:n, n]),
            jnp.asarray(np.squeeze(np.asarray(C))))


# ----------------------------------------------------------------------------
# config file loading (config/cidesi.toml)
# ----------------------------------------------------------------------------
# keys of the file that are not GliderParams fields: they are converted
_TF_KEYS = {"mass_tf": 2, "rot_pre_tf": 1, "rot_tf": 2, "vbd_pre_tf": 1,
            "vbd_tf": 2}                                   # -> n ZOH states
_DERIVED_KEYS = {"vbd_diam", "vbd_stroke"} | set(_TF_KEYS)
_DERIVED_FIELDS = {"vbd_area", "vbd_max_disp", "mass_A", "mass_B", "mass_C",
                   "rotpre_a", "rotpre_b", "rotpre_c", "rot_A", "rot_B",
                   "rot_C", "vbdpre_a", "vbdpre_b", "vbdpre_c", "vbd_A",
                   "vbd_B", "vbd_C"}
_INERTIA_3 = {"I_static", "I_lin", "I_rot", "I_pist"}     # 3 = diag, 3x3
_INERTIA_6 = {"MA", "MA_hull"}                            # 6 = diag, 6x6


def read_config(path=DEFAULT_CONFIG):
    """Raw TOML -> (flat parameter dict, {profile name: switches dict})."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    models = raw.pop("modelo", {})
    flat = {}
    for section, table in raw.items():
        for k, v in table.items():
            if k in flat:
                raise KeyError(f"{path}: '{k}' is repeated ([{section}])")
            flat[k] = v
    return flat, models


def _check_keys(where, given, expected):
    missing = sorted(set(expected) - set(given))
    unknown = sorted(set(given) - set(expected))
    if missing or unknown:
        raise KeyError(f"{where}: missing {missing}, unknown {unknown}")


def _as_matrix(name, v, n):
    a = jnp.asarray(v, dtype=float)
    if a.shape == (n,):
        return jnp.diag(a)
    if a.shape == (n, n):
        if not jnp.allclose(a, a.T):
            raise ValueError(f"{name} must be symmetric")
        return a
    raise ValueError(f"{name}: expected {n} values or {n}x{n}, got {a.shape}")


def params_from_dict(d, dt=0.05) -> GliderParams:
    """Flat dict (file keys) -> GliderParams; TFs discretized with dt."""
    expected = (set(GliderParams._fields) - _DERIVED_FIELDS) | _DERIVED_KEYS
    _check_keys("parameters", d, expected)
    out = {}
    for k, v in d.items():
        if k in _DERIVED_KEYS:
            continue
        if k in _INERTIA_3:
            out[k] = _as_matrix(k, v, 3)
        elif k in _INERTIA_6:
            out[k] = _as_matrix(k, v, 6)
        else:
            out[k] = jnp.asarray(v, dtype=float)
    out["vbd_area"] = jnp.asarray(math.pi * (d["vbd_diam"] / 2.0) ** 2)
    out["vbd_max_disp"] = jnp.asarray(d["vbd_stroke"] / 2.0)
    tfs = {}
    for k, n in _TF_KEYS.items():
        tf = d[k]
        num, den = (tf["num"], tf["den"]) if isinstance(tf, dict) else tf
        tfs[k] = zoh_tf(num, den, dt, n)
    out["mass_A"], out["mass_B"], out["mass_C"] = tfs["mass_tf"]
    out["rotpre_a"], out["rotpre_b"], out["rotpre_c"] = \
        [jnp.squeeze(v) for v in tfs["rot_pre_tf"]]
    out["rot_A"], out["rot_B"], out["rot_C"] = tfs["rot_tf"]
    out["vbdpre_a"], out["vbdpre_b"], out["vbdpre_c"] = \
        [jnp.squeeze(v) for v in tfs["vbd_pre_tf"]]
    out["vbd_A"], out["vbd_B"], out["vbd_C"] = tfs["vbd_tf"]
    return GliderParams(**out)


def load_params(path=DEFAULT_CONFIG, dt=0.05, **overrides) -> GliderParams:
    """Parameters from the config file. overrides use file keys, e.g.
    load_params(vbd_tf=([1.0], [4.0, 1.0])) or load_params(KM=-13.0)."""
    d, _ = read_config(path)
    for k in overrides:
        if k not in d:
            raise KeyError(f"unknown parameter '{k}'")
    d.update(overrides)
    return params_from_dict(d, dt)


def load_model(path=DEFAULT_CONFIG, model="simulink") -> ModelConfig:
    """Switch profile [modelo.<model>] of the config file."""
    _, models = read_config(path)
    if model not in models:
        raise KeyError(f"{path}: no profile [modelo.{model}]; "
                       f"available: {sorted(models)}")
    sw = models[model]
    _check_keys(f"[modelo.{model}]", sw,
                [f.name for f in dataclasses.fields(ModelConfig)])
    return ModelConfig(**sw)


def load_config(path=DEFAULT_CONFIG, model="simulink", dt=0.05):
    """(GliderParams, ModelConfig) from one config file."""
    return load_params(path, dt), load_model(path, model)


def cidesi_defaults(dt=0.05, **overrides) -> GliderParams:
    """Parameters of config/cidesi.toml (kept for the v1 scripts)."""
    return load_params(DEFAULT_CONFIG, dt, **overrides)


# backward-compat alias used by demos
seawing_like_defaults = cidesi_defaults
