"""run_test.py -- THE STARTING SCRIPT. Copy, edit the CONFIG section, run:

    python run_test.py        (from the gliderjax root folder)

Everything you can change is in the marked sections. Figures land in out/.
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the
# `gliderjax` package importable when running this file from the repo root.

import numpy as np
import jax.numpy as jnp
from gliderjax import ModelConfig, make_simulator, initial_state, scope
from gliderjax.params import cidesi_defaults

# =============================================================================
# 1) PARAMETERS  (the physics numbers: masses, coefficients, limits)
# =============================================================================
p = cidesi_defaults()          # values from config/cidesi.toml (edit there)

# To change a parameter only for this test, override here.
# GliderParams is immutable; ._replace() returns a modified copy. Examples:
#
# p = p._replace(disturb_xlin=0.005)             # 5 mm virtual-zero error
# p = p._replace(m_dist=0.5, r_dist=jnp.array([0.3, 0.0, 0.0]))  # 0.5 kg mass
# p = p._replace(current_n=jnp.array([0.1, -0.05, 0.0]))   # NED current [m/s]
# p = p._replace(KM=p.KM * 1.10)                # +10% pitch stiffness
#
# Full field list: config/cidesi.toml / gliderjax/params.py -- every
# NamedTuple field can be overridden this way, and this is also how Monte
# Carlo batching perturbs them.

# =============================================================================
# 2) MODEL CONFIGURATION  (structural switches -- change WHICH model runs)
#    Changing any of these triggers a ~5 s recompile. That is normal.
# =============================================================================
cfg = ModelConfig(
    kmr_mode="signum",
    #   "signum" -> KMR_mag * sign(alpha)   : EXACTLY your Simulink hack.
    #   "parity" -> KMR_mag + KMR_o*CLn(a)  : smooth research model (needs
    #                                         KMR_o from the +-alpha CFD).
    #   "const"  -> KMR_mag, no alpha dependence: the model that CANNOT
    #                                         invert the turn (for the paper).
    kmy_cubic=True,
    #   True  -> T3 = (15.04*b - 398.44*b^3)V^2 : your ANSYS fit (stiffness
    #            reverses at |beta| ~ 11 deg).
    #   False -> linear yaw stiffness only.
    use_ca_translational=False,
    #   True adds the analytic Munk block of C_A. This DOUBLE COUNTS physics
    #   already inside your ANSYS coefficients -- keep False except when
    #   deliberately demonstrating the phantom moment for the paper.
    use_ca_omega=False,
    #   True adds the omega-coupled (centripetal) C_A block built from
    #   MA_hull -- the ideal-flow rotation loads your STATIC CFD cannot
    #   contain. Keep False for Simulink validation (your Simulink nullifies
    #   all of C_A); revisit for spiral-accuracy studies.
    use_rot_damping=True,
    #   Your D_body rotational damping (4p+24p|p|, 10q+50q|q|, 100r|r|).
    #   False only for pure-potential-flow academic tests.
)

# =============================================================================
# 3) CONTROLLER  (what drives the actuators; runs at a fixed discrete rate)
#    Signature:  cmds, new_ctrl_state = ctrl(t, x, ctrl_state, params)
#    cmds = jnp.array([rc_cmd, gamma_cmd, dB_ref]):
#       rc_cmd    [m]   moving-mass position command (saturated to +-0.10,
#                       then through your 400/(s^2+30s+400) actuator TF)
#       gamma_cmd [rad] rotary-mass angle command (rate-limited 0.5 rad/s)
#       dB_ref    [N]   buoyancy command -> piston target via YOUR VBD block
#                       (saturates at +-4.93 N = full stroke), slew-limited.
#    ctrl_state: any pytree you want carried between calls (integrators...).
#    Open-loop test = ignore inputs and return constants. Time-scheduled
#    open loop = use jnp.where on t (NOT python if: t is a traced value).
#
#    REFERENCE SHAPING (closed loop): build once, carry state in ctrl_state:
#        REF0, REF_STEP = make_ref_pipeline(CTRL_DT,
#              pitch_rate=0.1, pitch_tf=([1],[5,1]),   # Simulink values
#              yaw_rate=0.03,  yaw_tf=([1],[5,1]),
#              bypass_roll=True,        # roll shaping OFF (as in Simulink);
#              bypass_all=False)        # True = feed trajectories directly
#        ...inside controller:
#        ref_s, shaped = REF_STEP(ref_s, jnp.array([th_ref, psi_ref, phi_ref]))
#    See demos/demo_closedloop.py for the full wiring. This is REFERENCE
#    smoothing -- separate from the ACTUATOR chains in params.
# =============================================================================
def ctrl(t, x, cs, prm):
    rc = 0.03                                   # mass 30 mm forward
    gm = jnp.deg2rad(0.0)                       # rotary mass angle
    dB = jnp.where(t < 300.0, -3.0, -3.0)       # constant dive (edit for
    #                                             apogee test: -3.0 -> +3.0)
    return jnp.array([rc, gm, dB]), cs


# =============================================================================
# 4) SIMULATOR SETTINGS  (numerics + timing)
# =============================================================================
DT = 0.05          # RK4 step [s]. 0.05 validated; the actuator TF costs
#                    nothing here because it is ZOH-exact at any dt.
DECIMATE = 20      # inner steps per telemetry sample: 20 -> 1 Hz scopes.
#                    Use 1-5 when inspecting fast transients (more memory).
CTRL_EVERY = 10    # controller period in inner steps: 10*DT = 0.5 s -- this
#                    IS the discrete rate your embedded code will run at.

sim = make_simulator(ctrl, ctrl_state0=jnp.zeros(1), cfg=cfg,
                     dt=DT, decimate=DECIMATE, ctrl_every=CTRL_EVERY)

# =============================================================================
# 5) INITIAL CONDITIONS
#    Convenience args cover the usual ones:
# =============================================================================
x0 = initial_state(depth=0.0, pitch=-0.2, roll=0.0, u=0.2)

# Full 23-state layout, for anything else via  x0 = x0.at[IDX].set(VALUE):
#   [0] x north   [1] y east    [2] z depth(+down)
#   [3] phi roll  [4] theta     [5] psi yaw            (rad)
#   [6] u surge   [7] v sway    [8] w heave            (RELATIVE vel, body)
#   [9] p         [10] q        [11] r                 (rad/s)
#   [12] m_rl     [13:15] mass TF states     rc    = mass_C @ x[13:15]
#   [15] g_pre    [16] g_rl     [17:19] rotary TF      gamma = rot_C @ x[17:19]
#   [19] v_pre    [20] v_rl     [21:23] VBD TF         dB_f  = vbd_C @ x[21:23]
# Actuator SPEEDS are set in config/cidesi.toml as continuous num/den, or
# here: cidesi_defaults(vbd_tf=([1.0],[4.0,1.0]))  ->  VBD becomes 1/(4s+1).
# Pre-setting an actuator: easiest is to run the sim a few frames with the
# desired constant command, or set the TF states so C@x equals the value
# (for the canonical 2-state form set x[14]=value/mass_C[1] etc. -- or just
# let the transient play out; both sims must then do the same).
#
# IMPORTANT for Simulink comparison: both sims must start actuators in the
# SAME position (default here: all neutral, controller moves them).

# =============================================================================
# 6) RUN
# =============================================================================
N_FRAMES = 400                     # duration = N_FRAMES * DECIMATE * DT [s]
tel = sim(p, x0, N_FRAMES)         # first call compiles (~5-10 s), reruns
#                                    with same cfg/shapes are ~instant.

print(f"final: depth {float(tel['eta'][-1,2]):7.1f} m | "
      f"pitch {float(tel['eta'][-1,4])*57.3:+6.1f} deg | "
      f"alpha {float(tel['alpha'][-1])*57.3:+5.2f} deg | "
      f"V {float(tel['V'][-1]):.3f} m/s")

# =============================================================================
# 7) VISUALIZATION  ("scopes")
#    tel is a dict of time series. Signal syntax: "name" or "name:index".
# =============================================================================
os.makedirs("out", exist_ok=True)

scope.list_signals(tel)   # print every available signal + shapes. Deriveds:
#                           pitch_deg, roll_deg, yaw_deg, alpha_deg,
#                           beta_deg, depth.

# (a) Ad-hoc stacked scope -- one axis per signal, like dropping Simulink
#     scopes on lines:
scope.scope(tel, ["pitch_deg", "alpha_deg", "V",
                  "actuators:0",        # rc (moving mass position)
                  "actuators:2",        # dxv (piston)
                  "tau_flow_body:4"],   # hydrodynamic PITCH moment [N m]
            title="my test", save="out/adhoc.png")

# (b) Moment-balance overlay -- THE debugging plot. For ONE body axis, every
#     moment contribution on the same axes: tau_flow_body (hydro),
#     tau_drot (rot. damping), tau_coriolis_rb, tau_CA, tau_restoring
#     (gravity+buoyancy). They must sum ~0 at equilibrium; whichever curve
#     is out of family is the physics term with the bug.
#     axis: 0=X force, 1=Y, 2=Z, 3=roll K, 4=pitch M, 5=yaw N.
scope.moments_breakdown(tel, axis=4, save="out/pitch_balance.png")
scope.moments_breakdown(tel, axis=5, save="out/yaw_balance.png")

# (c) All six components of any wrench/velocity vector in one figure:
scope.scope_vector6(tel, "tau_flow_body", save="out/tau_flow6.png")

# (d) XZ trajectory and the full standard debug set (10 figures):
scope.trajectory_xz(tel, save="out/traj_xz.png")
scope.default_scopes(tel, outdir="out", prefix="mytest_")

# =============================================================================
# Telemetry reference (everything recorded at each emitted sample):
#   eta(6), nu_r(6), nu(6), eta_dot(6), nu_r_dot(6)   states & derivatives
#   pos_n(3), vel_n(3), accel_n(3)                    inertial kinematics
#   alpha, beta, V                                    flow angles & speed
#   tau_flow_body(6)   hydrodynamic wrench in BODY frame
#   tau_flow_flow(6)   the same in FLOW frame: [-D, SF, -L, T1, T2, T3]
#   tau_drot(6)        rotational damping wrench
#   tau_coriolis_rb(6) -C_RB(nu)nu
#   tau_CA(6)          added-mass Coriolis (zero unless flags enable it)
#   tau_restoring(6)   weight+buoyancy wrench (physical sign, = -g(eta))
#   r_g(3), B, x_b     mass center (with dist), buoyancy, CB shift
#   actuators(3)=[rc, gamma, dxv],  cmds(3),  t
# =============================================================================
