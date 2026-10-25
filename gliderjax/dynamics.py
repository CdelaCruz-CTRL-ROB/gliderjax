"""6-DOF CIDESI glider dynamics (flow-frame damping formulation).

State layout (26,):
    eta   = x[0:6]    [x y z phi theta psi]        (NED, z down)
    nu_r  = x[6:12]   [u v w p q r]                (relative velocity, body)
    actuator chains (prefilter -> rate limiter -> main TF states):
        [12]    m_rl              mass rate-limiter memory
        [13:15] m_ss(2)           mass main TF;  rc    = mass_C @ x[13:15]
        [15]    g_pre             rotary prefilter state
        [16]    g_rl              rotary rate-limiter memory
        [17:19] g_ss(2)           rotary main TF; gamma = rot_C @ x[17:19]
        [19]    v_pre             VBD prefilter state
        [20]    v_rl              VBD rate-limiter memory [N]
        [21:23] v_ss(2)           VBD main TF;   dB_f  = vbd_C @ x[21:23]
                                  piston dxv = static VBD map of dB_f
    ecef  = x[23:26]  truth position, integrated only if cfg.use_geodesy
"""
from __future__ import annotations

import jax.numpy as jnp

from .params import GliderParams, ModelConfig, G0, RHO


def skew(a):
    return jnp.array([[0.0, -a[2], a[1]],
                      [a[2], 0.0, -a[0]],
                      [-a[1], a[0], 0.0]])


def R_body_to_ned(phi, th, psi):
    cph, sph = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(th), jnp.sin(th)
    cps, sps = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([
        [cps * cth, -sps * cph + cps * sth * sph,  sps * sph + cps * cph * sth],
        [sps * cth,  cps * cph + sph * sth * sps, -cps * sph + sth * sps * cph],
        [-sth,       cth * sph,                    cth * cph],
    ])


def T_euler(phi, th):
    cph, sph = jnp.cos(phi), jnp.sin(phi)
    cth, tth = jnp.cos(th), jnp.tan(th)
    return jnp.array([
        [1.0, sph * tth, cph * tth],
        [0.0, cph, -sph],
        [0.0, sph / cth, cph / cth],
    ])


def R_flow_to_body(alpha, beta):
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    return jnp.array([
        [ca * cb, -ca * sb, -sa],
        [sb,       cb,       0.0],
        [sa * cb, -sa * sb,  ca],
    ])


# ----------------------------------------------------------------------------
# VBD block: exact port of fcn_VBD_actuator (delta_x is a STATE, rate-limited
# in sim.py; here we map delta_x -> B, r_p, x_b)
# ----------------------------------------------------------------------------
def vbd_outputs(p: GliderParams, dxv):
    dV = -dxv * p.vbd_area
    # neutral at dB = 0 for ANY mass set (Simulink: W_total = 58*9.81 fixed)
    B_neutral = (p.m_static + p.m_lin + p.m_rot + p.m_pist) * G0
    B = B_neutral + dV * RHO * G0
    r_p = jnp.array([p.vbd_x0 + dxv, 0.0, 0.0])
    x_centroid = p.vbd_x0 + dxv / 2.0
    dB_act = dV * RHO * G0
    x_b = jnp.where(B > 0, dB_act * x_centroid / B, 0.0)
    return B, r_p, x_b


def dxv_from_dB_ref(p: GliderParams, dB_ref):
    """Command mapping of the VBD block: dB_ref -> saturated piston target."""
    dx_req = -(dB_ref / (RHO * G0)) / p.vbd_area
    return jnp.clip(dx_req, -p.vbd_max_disp, p.vbd_max_disp)


# ----------------------------------------------------------------------------
# four-mass geometry (port of Mbr+Ma fcn, docs/referencia_simulink/
# fcn_Mbr_Ma.m; rotating-mass inertia as R I R^T, see docs/MODELO.md 2.2)
# ----------------------------------------------------------------------------
def mass_geometry(p: GliderParams, rc, gamma, dxv):
    r_lin = p.r_lin0 + jnp.array([rc + p.disturb_xlin, 0.0, 0.0])
    r_rot = jnp.array([p.x_rot,
                       -p.R_rot * jnp.sin(gamma),
                        p.R_rot * jnp.cos(gamma)])
    _, r_p, _ = vbd_outputs(p, dxv)
    cg, sg = jnp.cos(gamma), jnp.sin(gamma)
    Rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])
    I_rot_r = Rx @ p.I_rot @ Rx.T          # rotates with r_rot = Rx [x,0,R]
    #                                        (Simulink has Rx^T I Rx: error)
    m_tot = p.m_static + p.m_lin + p.m_rot + p.m_pist
    r_g = (p.m_static * p.r_static + p.m_lin * r_lin
           + p.m_rot * r_rot + p.m_pist * r_p) / m_tot

    def about_CO(I_cm, m, r):
        S = skew(r)
        return I_cm - m * (S @ S)
    I_b = (about_CO(p.I_static, p.m_static, p.r_static)
           + about_CO(p.I_lin, p.m_lin, r_lin)
           + about_CO(I_rot_r, p.m_rot, r_rot)
           + about_CO(p.I_pist, p.m_pist, r_p))
    return m_tot, r_g, I_b


def M_rigid(m_tot, r_g, I_b):
    Sg = skew(r_g)
    top = jnp.concatenate([m_tot * jnp.eye(3), -m_tot * Sg], axis=1)
    bot = jnp.concatenate([m_tot * Sg, I_b], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def C_rigid(m_tot, r_g, I_b, nu):
    w = nu[3:6]
    Sw, Sg = skew(w), skew(r_g)
    top = jnp.concatenate([m_tot * Sw, -m_tot * Sw @ Sg], axis=1)
    bot = jnp.concatenate([m_tot * Sg @ Sw, -skew(I_b @ w)], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def C_added(MA, nu):
    v, w = nu[0:3], nu[3:6]
    A11, A12 = MA[0:3, 0:3], MA[0:3, 3:6]
    A21, A22 = MA[3:6, 0:3], MA[3:6, 3:6]
    Sv = skew(A11 @ v + A12 @ w)
    Sw_ = skew(A21 @ v + A22 @ w)
    Z = jnp.zeros((3, 3))
    top = jnp.concatenate([Z, -Sv], axis=1)
    bot = jnp.concatenate([-Sv, -Sw_], axis=1)
    return jnp.concatenate([top, bot], axis=0)


# ----------------------------------------------------------------------------
# Simulink surface model (exact port of fcn_buoyancy_surface): emergence of
# the hull top at nose and tail only, no roll. cfg.surface_model="simulink"
# ----------------------------------------------------------------------------
def surface_two_point(p: GliderParams, z, th):
    """Returns (B_scale, dxb_surface)."""
    Lh, H = p.hull_L / 2.0, p.hull_H
    sp, cp = jnp.sin(th), jnp.cos(th)
    e_nose = jnp.clip(-(z - Lh * sp - 0.5 * H * cp) / H, 0.0, 1.0)
    e_tail = jnp.clip(-(z + Lh * sp - 0.5 * H * cp) / H, 0.0, 1.0)
    avg = 0.5 * (e_nose + e_tail)
    e_sum = e_nose + e_tail
    dxb = jnp.where(e_sum > 1e-8,
                    -Lh * (e_nose - e_tail) / jnp.maximum(e_sum, 1e-8) * avg,
                    0.0)
    return 1.0 - avg, dxb


# ----------------------------------------------------------------------------
# free-surface transit model (port's replacement of the two-point model):
# 9-station immersion integral of the tilted hull -> buoyancy scale, CoB
# shift, added-mass scale, surface-proximity envelope; plus water-entry
# momentum flux and radiation damping (identify B33/B55/dmA_dz via a pool
# bobbing test: free decay from slight emergence, log depth vs time).
# ----------------------------------------------------------------------------
def surface_state(p: GliderParams, z, phi, th):
    """Immersion of a slender hull (length L along body x, height H).
    Returns (B_scale, dxb_surface, ma_scale, sigma_env)."""
    Lh, H = p.hull_L / 2.0, p.hull_H
    s = jnp.linspace(-1.0, 1.0, 9) * Lh                # body-x stations
    # NED depth of hull-TOP line at each station (z body-down, R31=-sin th,
    # R33=cos th cos phi)
    z_top = z - s * jnp.sin(th) - (H / 2.0) * jnp.cos(th) * jnp.cos(phi)
    emg = jnp.clip(-z_top / H, 0.0, 1.0)               # emerged fraction
    f = 1.0 - emg                                      # immersed fraction
    w_tr = jnp.ones(9).at[0].set(0.5).at[-1].set(0.5)  # trapezoid weights
    F = jnp.sum(w_tr * f)
    B_scale = F / jnp.sum(w_tr)
    x_cb = jnp.sum(w_tr * s * f) / jnp.maximum(F, 1e-6)
    dxb = jnp.where(B_scale > 1e-3, x_cb, 0.0)         # immersed-volume centroid
    ma_scale = jnp.maximum(B_scale, p.ma_scale_floor)
    # proximity envelope: 1 when any hull top at/above waterline, decays ~H
    h_top_min = jnp.min(z_top)
    sigma = jnp.exp(-jnp.maximum(h_top_min, 0.0) / H)
    return B_scale, dxb, ma_scale, sigma


def tau_surface(p: GliderParams, sigma, nu_r):
    """Localized surface forces: water-entry momentum flux (entry only) +
    heave/pitch radiation damping. Zero at depth (sigma -> 0)."""
    w, q = nu_r[2], nu_r[4]
    F_entry = -p.dmA_dz * jnp.maximum(w, 0.0) ** 2     # resists entry
    F_rad = -p.B33_rad * w
    M_rad = -p.B55_rad * q
    return sigma * jnp.array([0.0, 0.0, F_entry + F_rad, 0.0, M_rad, 0.0])


def tau_crossflow(p: GliderParams, nu_r):
    """Large-incidence quadratic crossflow drag, blended in by total
    incidence angle chi = acos(u/V). Near-zero in the glide regime
    (chi < ~20 deg); owns the physics beyond the ANSYS polynomial envelope."""
    u, v, w = nu_r[0], nu_r[1], nu_r[2]
    V = jnp.sqrt(u * u + v * v + w * w)
    chi = jnp.arccos(jnp.clip(jnp.abs(u) / jnp.maximum(V, 1e-6), 0.0, 1.0))
    wt = 0.5 * (1.0 + jnp.tanh((chi - p.chi_blend) / p.chi_width))
    return wt * jnp.array([0.0,
                           -p.Cd_cross_y * v * jnp.abs(v),
                           -p.Cd_cross_z * w * jnp.abs(w),
                           0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------
# hydrodynamics (exact port of the user's damping fcn)
# ----------------------------------------------------------------------------
def flow_angles(nu_r):
    u, v, w = nu_r[0], nu_r[1], nu_r[2]
    V = jnp.sqrt(u * u + v * v + w * w)
    Vs = jnp.maximum(V, 1e-6)
    alpha = jnp.arctan2(w, jnp.where(jnp.abs(u) < 1e-9, 1e-9, u))
    beta = jnp.arcsin(jnp.clip(v / Vs, -1.0, 1.0))
    return alpha, beta, V


def tau_flowframe(p: GliderParams, cfg: ModelConfig, nu_r):
    alpha, beta, V = flow_angles(nu_r)
    V2 = V * V
    if cfg.use_alpha_saturation:        # port guard, NOT in Simulink
        A_MAX = 0.44
        alpha_e = A_MAX * jnp.tanh(alpha / A_MAX)
        beta_e = A_MAX * jnp.tanh(beta / A_MAX)
    else:                               # Simulink: raw flow angles
        alpha_e, beta_e = alpha, beta

    if cfg.kmr_mode == "signum":        # Simulink: KMR * sign(alpha)
        KMR = p.KMR_mag * jnp.sign(alpha)
    elif cfg.kmr_mode == "parity":      # research variant: K_e + K_o*CLn
        CLn = (p.KL0 + p.KL * alpha_e) / p.KL
        KMR = p.KMR_mag + p.KMR_o * CLn
    else:                               # "const"
        KMR = p.KMR_mag

    D = (p.KD0 + p.KD * alpha_e ** 2) * V2
    SF = p.Kbeta * beta_e * V2
    L = (p.KL0 + p.KL * alpha_e) * V2
    T1 = KMR * beta_e * V2
    T2 = (p.KM0 + p.KM * alpha_e) * V2
    if cfg.kmy_cubic:
        T3 = (p.KMY_lin * beta_e + p.KMY_cub * beta_e ** 3) * V2
    else:
        T3 = p.KMY_lin * beta_e * V2

    tau_f = jnp.array([-D, SF, -L, T1, T2, T3])
    Rfb = R_flow_to_body(alpha, beta)
    tau_b = jnp.concatenate([Rfb @ tau_f[0:3], Rfb @ tau_f[3:6]])
    return tau_b, tau_f, alpha, beta, V


def tau_rot_damping(p: GliderParams, cfg: ModelConfig, nu_r, V):
    if not cfg.use_rot_damping:
        return jnp.zeros(6)
    pr, qr, rr = nu_r[3], nu_r[4], nu_r[5]
    Mx = -(p.Kp_lin + p.Kp_quad * jnp.abs(pr)) * pr - p.KpV * V * pr
    My = -(p.Kq_lin + p.Kq_quad * jnp.abs(qr)) * qr - p.KqV * V * qr
    Mz = -(p.Kr_lin + p.Kr_quad * jnp.abs(rr)) * rr - p.KrV * V * rr
    return jnp.array([0.0, 0.0, 0.0, Mx, My, Mz])


def tau_restoring(p: GliderParams, eta, r_g_dist, m_tot, B, x_b):
    """Physical wrench (+). Equals -(report g(eta)). Uses DISTURBED r_g and
    the VBD-shifted buoyancy application point [x_b, 0, 0]."""
    phi, th, psi = eta[3], eta[4], eta[5]
    W = m_tot * G0
    Rnb = R_body_to_ned(phi, th, psi).T
    f_g = Rnb @ jnp.array([0.0, 0.0, W])
    f_b = Rnb @ jnp.array([0.0, 0.0, -B])
    r_b = jnp.array([x_b, 0.0, 0.0])
    force = f_g + f_b
    moment = jnp.cross(r_g_dist, f_g) + jnp.cross(r_b, f_b)
    return jnp.concatenate([force, moment])


# ----------------------------------------------------------------------------
# assembled dynamics
# ----------------------------------------------------------------------------
def dynamics(x, p: GliderParams, cfg: ModelConfig):
    eta, nu_r = x[0:6], x[6:12]
    rc = p.mass_C @ x[13:15]
    gamma = p.rot_C @ x[17:19]
    dB_f = p.vbd_C @ x[21:23]
    dxv = dxv_from_dB_ref(p, dB_f)        # static VBD map (incl. stroke sat)

    m_tot, r_g, I_b = mass_geometry(p, rc, gamma, dxv)
    B, r_p, x_b = vbd_outputs(p, dxv)

    if cfg.surface_model == "full":
        B_scale, dxb_s, ma_scale, sigma = surface_state(p, eta[2], eta[3], eta[4])
    elif cfg.surface_model == "simulink":
        B_scale, dxb_s = surface_two_point(p, eta[2], eta[4])
        ma_scale, sigma = 1.0, 0.0
    else:
        B_scale, dxb_s, ma_scale, sigma = 1.0, 0.0, 1.0, 0.0
    B_eff = B * B_scale
    x_b_eff = x_b + dxb_s      # VBD shift + surface shift (simple sum, as Simulink)

    # disturbance wiring replication: disturb_xg/yg enter restoring + C_RB
    # via r_g, but NOT M_RB (the Simulink builds M from undisturbed positions).
    r_g_dist = r_g + jnp.array([p.disturb_xg, p.disturb_yg, 0.0])
    r_g_M = r_g if cfg.replicate_disturb_wiring else r_g_dist
    r_g_C = r_g_dist

    MRB = M_rigid(m_tot, r_g_M, I_b)
    M = MRB + p.MA * ma_scale

    tau_hs = tau_restoring(p, eta, r_g_dist, m_tot, B_eff, x_b_eff)
    tau_fb, tau_ff, alpha, beta, V = tau_flowframe(p, cfg, nu_r)
    tau_dr = tau_rot_damping(p, cfg, nu_r, V)
    tau_sf = (tau_surface(p, sigma, nu_r) if cfg.surface_model == "full"
              else jnp.zeros(6))
    tau_cf = tau_crossflow(p, nu_r) if cfg.use_crossflow else jnp.zeros(6)
    tau_crb = -C_rigid(m_tot, r_g_C, I_b, nu_r) @ nu_r

    tau_ca = jnp.zeros(6)
    if cfg.use_ca_omega or cfg.use_ca_translational:
        CA_full = C_added(p.MA_hull, nu_r) @ nu_r
        nu_t = nu_r.at[3:6].set(0.0)
        CA_trans = C_added(p.MA_hull, nu_t) @ nu_t
        CA_omega = CA_full - CA_trans
        if cfg.use_ca_omega:
            tau_ca = tau_ca - CA_omega
        if cfg.use_ca_translational:
            tau_ca = tau_ca - CA_trans

    rhs = tau_fb + tau_dr + tau_sf + tau_cf + tau_crb + tau_ca + tau_hs
    nu_r_dot = jnp.linalg.solve(M, rhs)

    Rbn = R_body_to_ned(eta[3], eta[4], eta[5])
    nu_c = jnp.concatenate([Rbn.T @ p.current_n, jnp.zeros(3)])
    nu = nu_r + nu_c
    eta_dot = jnp.concatenate([Rbn @ nu[0:3], T_euler(eta[3], eta[4]) @ nu[3:6]])
    a_n = Rbn @ (nu_r_dot[0:3] + jnp.cross(nu[3:6], nu[0:3]))

    if cfg.use_geodesy:
        from .geodesy import ecef2lla, R_n_e
        lla = ecef2lla(x[23:26], iters=2)
        ecef_dot = R_n_e(lla[0], lla[1]) @ eta_dot[0:3]
    else:
        ecef_dot = jnp.zeros(3)
    x_dot = jnp.concatenate([eta_dot, nu_r_dot, jnp.zeros(11), ecef_dot])

    tel = {
        "eta": eta, "nu_r": nu_r, "nu": nu,
        "eta_dot": eta_dot, "nu_r_dot": nu_r_dot,
        "accel_n": a_n, "vel_n": eta_dot[0:3], "pos_n": eta[0:3],
        "alpha": alpha, "beta": beta, "V": V,
        "tau_flow_body": tau_fb, "tau_flow_flow": tau_ff,
        "tau_drot": tau_dr, "tau_coriolis_rb": tau_crb,
        "tau_CA": tau_ca, "tau_restoring": tau_hs,
        "r_g": r_g_dist, "B": B_eff, "x_b": x_b_eff,
        "B_net": B_eff - m_tot * G0,      # net buoyancy [N], + = floats
        "B_scale": jnp.asarray(B_scale),
        "sigma_surface": jnp.asarray(sigma),
        "tau_surface": tau_sf, "tau_crossflow": tau_cf,
        "ecef": x[23:26],
        "actuators": jnp.array([rc, gamma, dxv]),
    }
    return x_dot, tel
