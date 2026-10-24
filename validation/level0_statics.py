"""Level 0: statics -- no hydrodynamic coefficients exercised beyond damping.
T1 pitch pendulum | T2 roll pendulum | T3 gamma-step steady roll
T4 passivity (coast-down) | T5 glide (Level 1a preview)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax.numpy as jnp
from gliderjax import ModelConfig, make_simulator, initial_state, scope
from gliderjax.params import cidesi_defaults

p = cidesi_defaults()
cfg = ModelConfig()
DT, DEC = 0.05, 20

def const_ctrl(rc=0.0, gm=0.0, dB=0.0):
    def ctrl(t, x, cs, params):
        return jnp.array([rc, gm, dB]), cs
    return ctrl

def run(ctrl, x0, T):
    sim = make_simulator(ctrl, jnp.zeros(1), cfg, dt=DT, decimate=DEC, ctrl_every=10)
    tel = sim(p, x0, int(T/(DT*DEC)))
    return tel

out = os.path.join(os.path.dirname(__file__), "..", "out")

# T1 pitch pendulum: theta0=10deg, everything neutral
tel = run(const_ctrl(), initial_state(pitch=np.deg2rad(10)), 120)
th = np.asarray(tel["eta"])[:,4]*57.3
print(f"T1 pitch pendulum: theta0=10deg -> final {th[-1]:+.3f} deg "
      f"(expect ~0), max overshoot {th.min():+.2f}")
scope.scope(tel, ["pitch_deg","nu_r:4","tau_restoring:4","tau_drot:4"],
            title="T1 pitch pendulum", save=os.path.join(out,"L0_T1_pitch.png"))

# T2 roll pendulum
tel = run(const_ctrl(), initial_state(roll=np.deg2rad(10)), 120)
ph = np.asarray(tel["eta"])[:,3]*57.3
print(f"T2 roll pendulum:  phi0=10deg  -> final {ph[-1]:+.3f} deg (expect ~0)")
scope.scope(tel, ["roll_deg","nu_r:3","tau_restoring:3","tau_drot:3"],
            title="T2 roll pendulum", save=os.path.join(out,"L0_T2_roll.png"))

# T3 gamma step +30deg: predicted steady roll = atan(y_g/z_g)
tel = run(const_ctrl(gm=np.deg2rad(30)), initial_state(), 150)
ph = np.asarray(tel["eta"])[:,3]*57.3
m_tot = 58.0
y_g = -5.0*0.07*np.sin(np.deg2rad(30))/m_tot
# z_g at gamma=30: static+lin z + rot z cos(gamma) + piston 0
z_g = (37*0.001 + 15*0.001 + 5*0.07*np.cos(np.deg2rad(30)))/m_tot
print(f"T3 gamma=+30deg: steady roll {ph[-1]:+.2f} deg "
      f"(prediction atan(y_g/z_g) = {np.rad2deg(np.arctan(y_g/z_g)):+.2f} deg)")
scope.scope(tel, ["roll_deg","actuators:1","tau_restoring:3"],
            title="T3 gamma step", save=os.path.join(out,"L0_T3_gamma.png"))

# T4 passivity: initial speed, no forcing, neutral B -> V decays monotonically
tel = run(const_ctrl(), initial_state(u=0.4), 200)
V = np.asarray(tel["V"])
mono = np.all(np.diff(V) < 1e-9)
print(f"T4 passivity: V 0.4 -> {V[-1]:.4f} m/s, monotone decay: {mono}")

# T5 glide preview: mass fwd 30mm, dB=-3N
tel = run(const_ctrl(rc=0.03, dB=-3.0), initial_state(pitch=-0.2, u=0.2), 400)
e = np.asarray(tel["eta"])
print(f"T5 glide: theta_ss {e[-1,4]*57.3:+.1f} deg, alpha "
      f"{float(np.asarray(tel['alpha'])[-1])*57.3:+.2f} deg, "
      f"V {float(np.asarray(tel['V'])[-1]):.3f} m/s, "
      f"glide ratio {e[-1,0]/e[-1,2]:.2f}, depth {e[-1,2]:.1f} m")
scope.default_scopes(tel, outdir=out, prefix="L0_T5_glide_")
