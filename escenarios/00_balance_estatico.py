"""00_balance_estatico.py -- balance estático del planeador, SIN simular.

Con los parámetros del archivo de configuración calcula:
  - masas, posiciones, peso, flotabilidad y flotabilidad neta;
  - volumen desplazado necesario para ser neutro;
  - centro de gravedad (CG), centro de flotación (CB) y BG;
  - CERO VIRTUAL de la masa móvil: dónde debe estar para pitch 0 en reposo,
    y el gamma de la masa rotatoria para roll 0. El cero NOMINAL se calcula
    sin la masa de perturbación (m_dist); aparte se muestra su efecto y el
    error de cero equivalente;
  - actitud de equilibrio en reposo (sin flujo) para unos actuadores dados;
  - curvas: pitch en reposo vs rc y roll en reposo vs gamma.

"En reposo" = solo peso y flotación (sin velocidad, sin hidrodinámica). En
planeo el pitch es otro: se suma el momento hidrodinámico.

Correr desde GLIDER_JAX\\sim\\gliderjax:

    ..\\Scripts\\python.exe escenarios\\00_balance_estatico.py
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)          # para importar la librería gliderjax

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
import jax
import jax.numpy as jnp
from gliderjax import scope
from gliderjax.params import load_params, G0
from gliderjax.dynamics import (mass_geometry, vbd_outputs, dxv_from_dB_ref,
                                tau_restoring, disturbance_zero_equivalent)

# =============================================================================
# 1) CONFIGURACIÓN
# =============================================================================
CONFIG = os.path.join(RAIZ, "config", "cidesi.toml")

# =============================================================================
# 2) ACTUADORES A EVALUAR (posiciones ya alcanzadas)
# =============================================================================
RC = 0.0              # [m]   masa móvil (+ = adelante)
GAMMA_DEG = 0.0       # [deg] masa rotatoria
DB = 0.0              # [N]   VBD (+ = sube)

# curvas
RC_CURVA = np.linspace(-0.10, 0.10, 41)         # [m]
GAMMA_CURVA_DEG = np.linspace(-90.0, 90.0, 37)  # [deg]
DB_CURVA = (-3.0, 0.0, 3.0)                     # [N] una curva de pitch por valor

MOSTRAR = True

# =============================================================================
# 3) CÁLCULO
# =============================================================================
NOMBRE = "00_balance_estatico"
OUT = os.path.join(RAIZ, "out", NOMBRE)
os.makedirs(OUT, exist_ok=True)
p = load_params(CONFIG)
p_nom = p._replace(m_dist=0.0 * p.m_dist)
perturbado = bool(float(p.m_dist))
gamma = np.deg2rad(GAMMA_DEG)


@jax.jit
def _momentos(q, phi, th, rc, gm, dB):
    dxv = dxv_from_dB_ref(q, dB)
    m, r_g, _ = mass_geometry(q, rc, gm, dxv)
    B, _, x_b = vbd_outputs(q, dxv)
    eta = jnp.array([0.0, 0.0, 0.0, phi, th, 0.0])
    return tau_restoring(q, eta, r_g, m, B, x_b)[3:5]


def momentos(phi, th, rc, gm, dB, q=p):
    """Momentos de restauración [Mx, My] en reposo con actitud (phi, th)."""
    return _momentos(q, phi, th, rc, gm, dB)


def equilibrio(rc, gm, dB, q=p):
    """(roll, pitch) [rad] del equilibrio ESTABLE en reposo (|ángulos| < 90°):
    momentos de restauración = 0. Punto de partida: pitch de la solución 1-D
    con roll 0, y roll ~ atan(y_g / z_g)."""
    lim = np.pi / 2 - 1e-6
    My = lambda th: float(momentos(0.0, th, rc, gm, dB, q)[1])
    th0 = brentq(My, -lim, lim) if My(-lim) * My(lim) < 0 else 0.0
    dxv = dxv_from_dB_ref(q, dB)
    _, r_g, _ = mass_geometry(q, rc, gm, dxv)
    phi0 = np.arctan2(float(r_g[1]), float(r_g[2]))
    f = lambda a: np.asarray(momentos(a[0], a[1], rc, gm, dB, q))
    sol, _, ok, _ = fsolve(f, (phi0, th0), full_output=True)
    sol = (np.asarray(sol) + np.pi) % (2 * np.pi) - np.pi
    if ok != 1 or np.any(np.abs(sol) > np.pi / 2):
        return np.array([np.nan, np.nan])
    return sol


def geometria(rc, gm, dB):
    dxv = float(dxv_from_dB_ref(p, dB))
    m, r_g, _ = mass_geometry(p, rc, gm, dxv)
    B, r_p, x_b = vbd_outputs(p, dxv)
    r_lin = np.asarray(p.r_lin0) + np.array([rc, 0, 0])
    r_rot = np.array([float(p.x_rot), -float(p.R_rot) * np.sin(gm),
                      float(p.R_rot) * np.cos(gm)])
    return dict(m=float(m), r_g=np.asarray(r_g), B=float(B), x_b=float(x_b),
                dxv=dxv, r_lin=r_lin, r_rot=r_rot, r_p=np.asarray(r_p))


g = geometria(RC, gamma, DB)
W = g["m"] * G0
r_g = g["r_g"]
B_max = float(p.rho) * G0 * float(p.vbd_area) * float(p.vbd_max_disp)

print(f"configuración: {os.path.relpath(CONFIG, RAIZ)}")
print(f"actuadores: rc = {RC * 1000:+.1f} mm, gamma = {GAMMA_DEG:+.1f}°, "
      f"dB = {DB:+.2f} N (pistón {g['dxv'] * 1000:+.1f} mm)")
if perturbado:
    print(f"masa de perturbación: {float(p.m_dist):.3f} kg en "
          f"{np.asarray(p.r_dist)} m")
print()

print("== MASAS ==")
print(f"{'':12s}{'m [kg]':>9s}{'x [m]':>10s}{'y [m]':>10s}{'z [m]':>10s}")
for nombre, m_i, r_i in (("estática", p.m_static, p.r_static),
                         ("móvil", p.m_lin, g["r_lin"]),
                         ("rotatoria", p.m_rot, g["r_rot"]),
                         ("pistón", p.m_pist, g["r_p"]),
                         ("perturb.", p.m_dist, p.r_dist)):
    if nombre == "perturb." and not float(m_i):
        continue
    r_i = np.asarray(r_i)
    print(f"{nombre:12s}{float(m_i):9.3f}{r_i[0]:10.4f}{r_i[1]:10.4f}{r_i[2]:10.4f}")
print(f"{'total':12s}{g['m']:9.3f}   peso W = {W:.3f} N")

print("\n== FLOTABILIDAD ==")
V_neutro = g["m"] / float(p.rho)
B_neutral = float(p.rho) * G0 * float(p.V_displaced)
print(f"rho = {float(p.rho):.1f} kg/m^3,  V_displaced = {float(p.V_displaced):.6f} m^3")
print(f"B con dB = 0: {B_neutral:.3f} N;  con dB actual: {g['B']:.3f} N")
print(f"flotabilidad neta B - W: dB = 0 -> {B_neutral - W:+.3f} N;  "
      f"dB actual -> {g['B'] - W:+.3f} N   (+ = flota)")
print(f"V_displaced para ser neutro con dB = 0: {V_neutro:.6f} m^3 "
      f"(diferencia {(V_neutro - float(p.V_displaced)) * 1000:+.3f} L)")
print(f"rango del VBD: ±{B_max:.3f} N", end="")
if abs(B_neutral - W) > B_max:
    print("  -> NO alcanza a compensar la diferencia de neutralidad")
else:
    print(f"  -> neutro con dB = {W - B_neutral:+.3f} N")

print("\n== CENTROS (marco cuerpo, origen = CB con VBD neutro) ==")
print(f"CG: x_g = {r_g[0] * 1000:+.3f} mm, y_g = {r_g[1] * 1000:+.3f} mm, "
      f"z_g = {r_g[2] * 1000:+.3f} mm" + ("   (con perturbaciones)" if perturbado else ""))
print(f"CB: x_b = {g['x_b'] * 1000:+.3f} mm")
print(f"BG (z_g - z_b) = {r_g[2] * 1000:+.3f} mm  "
      f"({'CG abajo del CB: estable' if r_g[2] > 0 else 'CG ARRIBA del CB: INESTABLE'})")

print("\n== CERO VIRTUAL NOMINAL (dB = 0, sin perturbaciones) ==")
# rc para pitch 0 en reposo con gamma = 0: My(0, 0) = 0, lineal en rc
rc_pitch0 = lambda q: brentq(
    lambda rc: float(momentos(0.0, 0.0, rc, 0.0, 0.0, q)[1]), -1.0, 1.0)
rc0 = rc_pitch0(p_nom)
x_fis = float(p.r_lin0[0]) + rc0
print(f"masa móvil: para pitch 0 debe estar en x = {x_fis:.5f} m")
print(f"  r_lin0 del archivo = {float(p.r_lin0[0]):.5f} m  ->  rc para pitch 0 = "
      f"{rc0 * 1000:+.3f} mm")
th_rc0 = equilibrio(0.0, 0.0, 0.0, p_nom)[1]
print(f"  con rc = 0, pitch en reposo = {np.rad2deg(th_rc0):+.3f}°")
Mx0 = lambda gm, q: float(momentos(0.0, 0.0, rc0, gm, 0.0, q)[0])
try:
    gm0 = brentq(lambda gm: Mx0(gm, p_nom), -np.pi / 2, np.pi / 2)
    print(f"masa rotatoria: gamma para roll 0 = {np.rad2deg(gm0):+.3f}°")
except ValueError:
    print("masa rotatoria: ningún gamma en ±90° da roll 0")

if perturbado:
    print("\n== EFECTO DE LA MASA DE PERTURBACIÓN (el control no la conoce) ==")
    dx, dg = disturbance_zero_equivalent(p)
    print(f"error de cero equivalente: masa móvil {float(dx) * 1000:+.3f} mm "
          f"(+ = más adelante de lo que cree el control), gamma "
          f"{np.rad2deg(float(dg)):+.3f}°")
    rc0_p = rc_pitch0(p)
    th_p = equilibrio(0.0, 0.0, 0.0)[1]
    print(f"rc para pitch 0: {rc0 * 1000:+.3f} mm nominal -> {rc0_p * 1000:+.3f} mm "
          f"real ({(rc0_p - rc0) * 1000:+.3f} mm)")
    print(f"pitch en reposo con rc = 0: {np.rad2deg(th_rc0):+.3f}° nominal -> "
          f"{np.rad2deg(th_p):+.3f}° real")
    try:
        gm0_p = brentq(lambda gm: float(momentos(0.0, 0.0, rc0_p, gm, 0.0)[0]),
                       -np.pi / 2, np.pi / 2)
        print(f"gamma para roll 0: {np.rad2deg(gm0):+.3f}° nominal -> "
              f"{np.rad2deg(gm0_p):+.3f}° real")
    except ValueError:
        print("gamma para roll 0: ningún gamma en ±90° (la masa rotatoria no "
              "alcanza a compensar la perturbación)")

print(f"\n== EQUILIBRIO EN REPOSO con los actuadores de la sección 2 ==")
phi_e, th_e = equilibrio(RC, gamma, DB)
print(f"roll = {np.rad2deg(phi_e):+.3f}°, pitch = {np.rad2deg(th_e):+.3f}°")

# ---- curvas -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for dB in DB_CURVA:
    th = [equilibrio(rc, 0.0, dB)[1] for rc in RC_CURVA]
    ax[0].plot(RC_CURVA * 1000, np.rad2deg(th), label=f"dB = {dB:+.1f} N")
ax[0].axhline(0, color="0.6", lw=0.8)
ax[0].axvline(rc0 * 1000, color="0.6", lw=0.8, ls="--")
ax[0].set_xlabel("rc [mm]")
ax[0].set_ylabel("pitch en reposo [deg]")
ax[0].set_title("Pitch en reposo vs masa móvil (gamma = 0)")
ax[0].legend()
ph = [equilibrio(RC, np.deg2rad(gd), DB)[0] for gd in GAMMA_CURVA_DEG]
ax[1].plot(GAMMA_CURVA_DEG, np.rad2deg(ph))
ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].set_xlabel("gamma [deg]")
ax[1].set_ylabel("roll en reposo [deg]")
ax[1].set_title(f"Roll en reposo vs masa rotatoria (rc = {RC * 1000:+.0f} mm)")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "curvas_reposo.png"), dpi=120)
print(f"\nfigura en {OUT}")
if MOSTRAR:
    plt.show()
