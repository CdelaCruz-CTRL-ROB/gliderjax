"""Glider parameters (JAX pytree) and static model configuration.

CIDESI-UNINDETEC parameter set transcribed from the user's Simulink blocks
(Mbr+Ma fcn, VBD actuator fcn, damping fcn). rho=1000 (freshwater) per the
VBD block -- confirm consistency with CFD density.
"""
from __future__ import annotations

from typing import NamedTuple
import dataclasses
import jax.numpy as jnp

G0 = 9.81
RHO = 1000.0          # per VBD block (freshwater)


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
    # ----- added mass (off-diagonals zeroed; leftovers removed) ------------
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
    dist_xg: jnp.ndarray       # parametric CG disturbance (restoring+coriolis
    dist_yg: jnp.ndarray       #  only, replicating the Simulink wiring)
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
    use_surface_model: bool = True      # immersion scaling + entry + radiation
    use_crossflow: bool = True          # large-incidence quadratic drag blend
    use_geodesy: bool = True            # integrate ECEF truth position
    replicate_dist_wiring: bool = True  # dist enters C_RB+restoring, NOT M_RB


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


def cidesi_defaults(dt=0.05,
        # -------- actuator chains: num/den EXACTLY as in the Simulink blocks
        mass_rate=0.008,                             # [m/s]
        mass_tf=([400.0], [1.0, 30.0, 400.0]),       # 400/(s^2+30s+400)
        rot_pre_tf=([1.0], [0.1, 1.0]),              # 1/(0.1s+1) helper;
        #                                              ([1],[1e-9,1]) ~= OFF
        rot_rate=0.5,                                # [rad/s]   <-- CONFIRM
        rot_tf=([400.0], [1.0, 30.0, 400.0]),        # 400/(s^2+30s+400)
        vbd_pre_tf=([1.0], [0.2, 1.0]),              # 1/(0.2s+1) helper
        vbd_rate_N=0.5,                              # [N/s]     <-- CONFIRM
        vbd_tf=([1.0], [2.0, 1.0]),                  # 1/(2s+1)  <-- CONFIRM
        ) -> GliderParams:
    mA, mB, mC = zoh_tf(*mass_tf, dt, 2)
    rpa, rpb, rpc = [jnp.squeeze(v) for v in zoh_tf(*rot_pre_tf, dt, 1)]
    rA, rB, rC = zoh_tf(*rot_tf, dt, 2)
    vpa, vpb, vpc = [jnp.squeeze(v) for v in zoh_tf(*vbd_pre_tf, dt, 1)]
    vA, vB, vC = zoh_tf(*vbd_tf, dt, 2)
    return GliderParams(
        m_static=jnp.asarray(37.0),
        r_static=jnp.array([-0.1241, 0.0, 0.001]),
        I_static=jnp.diag(jnp.array([0.9, 21.0, 21.0])),
        m_lin=jnp.asarray(15.0),
        r_lin0=jnp.array([0.2861, 0.0, 0.001]),
        I_lin=jnp.diag(jnp.array([0.02, 10.16, 0.17])),
        dx_lin_max=jnp.asarray(0.10),
        m_rot=jnp.asarray(5.0),
        x_rot=jnp.asarray(0.2),
        R_rot=jnp.asarray(0.07),
        I_rot=jnp.diag(jnp.array([0.00241296, 0.00207504, 0.00422813])),
        gamma_max=jnp.asarray(jnp.pi / 2),
        w_rot_max=jnp.asarray(0.5),  # rotary RATE limiter [rad/s]  # <-- CONFIRM
        m_pist=jnp.asarray(1.0),
        I_pist=jnp.diag(jnp.array([4.5521678e-4, 1.245908523e-2, 1.245908523e-2])),
        vbd_area=jnp.asarray(3.14159265 * 0.04 ** 2),
        vbd_max_disp=jnp.asarray(0.100),
        vbd_x0=jnp.asarray(-0.7),
        vbd_rate=jnp.asarray(0.01),
        mass_rate=jnp.asarray(mass_rate), mass_A=mA, mass_B=mB, mass_C=mC,
        rotpre_a=rpa, rotpre_b=rpb, rotpre_c=rpc,
        rot_rate=jnp.asarray(rot_rate), rot_A=rA, rot_B=rB, rot_C=rC,
        vbdpre_a=vpa, vbdpre_b=vpb, vbdpre_c=vpc,
        vbd_rate_N=jnp.asarray(vbd_rate_N), vbd_A=vA, vbd_B=vB, vbd_C=vC,
        MA=jnp.diag(jnp.array([3.9112, 58.17, 48.433, 0.5, 15.83, 15.83])),
        MA_hull=jnp.diag(jnp.array([3.9112, 45.0, 40.0, 0.1, 12.0, 12.0])),
        KD0=jnp.asarray(5.9), KD=jnp.asarray(214.62),
        KL0=jnp.asarray(0.0), KL=jnp.asarray(308.9754),
        KM0=jnp.asarray(-0.0328), KM=jnp.asarray(-12.48),
        Kbeta=jnp.asarray(-56.66),
        KMR_mag=jnp.asarray(4.5194), KMR_o=jnp.asarray(0.0),
        KMY_lin=jnp.asarray(15.04), KMY_cub=jnp.asarray(-398.442),
        Kp_lin=jnp.asarray(4.0), Kq_lin=jnp.asarray(10.0), Kr_lin=jnp.asarray(0.0),
        Kp_quad=jnp.asarray(24.0), Kq_quad=jnp.asarray(50.0), Kr_quad=jnp.asarray(100.0),
        KpV=jnp.asarray(0.0), KqV=jnp.asarray(0.0), KrV=jnp.asarray(0.0),
        hull_L=jnp.asarray(1.8), hull_H=jnp.asarray(0.2),
        B33_rad=jnp.asarray(120.0), B55_rad=jnp.asarray(40.0),
        dmA_dz=jnp.asarray(120.0), ma_scale_floor=jnp.asarray(0.3),
        Cd_cross_y=jnp.asarray(200.0), Cd_cross_z=jnp.asarray(200.0),
        chi_blend=jnp.asarray(0.40), chi_width=jnp.asarray(0.12),
        gps_sigma_m=jnp.asarray(0.0), gps_gate_depth=jnp.asarray(0.7),
        dist_xg=jnp.asarray(0.0), dist_yg=jnp.asarray(0.0),
        current_n=jnp.zeros(3),
    )


# backward-compat alias used by demos
seawing_like_defaults = cidesi_defaults
