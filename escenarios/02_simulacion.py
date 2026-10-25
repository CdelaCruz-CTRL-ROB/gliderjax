"""02_simulacion.py -- simulación por tiempo, para probar a mano.

Flujo de trabajo: cambias valores aquí (o en config/cidesi.toml), corres,
ves las gráficas, cierras, cambias, vuelves a correr. Cada corrida se
compara con la anterior: imprime qué cambió y superpone las dos.

Correr desde GLIDER_JAX\\sim\\gliderjax:

    ..\\Scripts\\python.exe escenarios\\02_simulacion.py

Resultados en out/02_simulacion/ (la corrida anterior en anterior/).
Documentación: docs/SIMULACION.md
"""
import json
import os
import shutil
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)          # para importar la librería gliderjax

import numpy as np
import matplotlib.pyplot as plt
from gliderjax import make_simulator, build_initial_state, scope, SOLVERS
from gliderjax.params import load_config, read_config, G0
from gliderjax.autopilot import make_open_loop_controller

# =============================================================================
# 1) MODELO  (parámetros y switches: editar config/cidesi.toml)
# =============================================================================
CONFIG = os.path.join(RAIZ, "config", "cidesi.toml")
MODELO = "simulink"   # perfil [modelo.<nombre>]: "simulink" | "completo"

# =============================================================================
# 2) SOLVER
# =============================================================================
SOLVER = "ode4"       # "ode4" Runge-Kutta 4 | "ode2" Heun | "ode1" Euler
#                       (paso fijo, mismos nombres que en Simulink)
DT = 0.005            # [s] paso de integración
T_FINAL = 600.0       # [s] duración
MUESTREO = 0.1        # [s] cada cuánto se guarda una muestra (múltiplo de DT)

# =============================================================================
# 3) CONDICIONES INICIALES
# =============================================================================
# --- en el marco inercial (NED: x norte, y este, z abajo) ---
POSICION = (0.0, 0.0, 50.0)          # [m]   x, y, z (profundidad)
ACTITUD_DEG = (0.0, 0.0, 0.0)        # [deg] roll, pitch, yaw
# --- velocidad lineal: en el cuerpo o en el inercial ---
VELOCIDAD_MARCO = "cuerpo"           # "cuerpo":   (u, v, w) relativa al agua
#                                      "inercial": (vN, vE, vD) respecto al fondo
VELOCIDAD = (0.0, 0.0, 0.0)          # [m/s]
# --- velocidad angular, en el cuerpo ---
VEL_ANGULAR_DEG = (0.0, 0.0, 0.0)    # [deg/s] p, q, r
# --- origen geográfico: solo si el perfil usa geodesia (use_geodesy) ---
ORIGEN_GEO = (-95.72275, 19.37214)   # [deg] lon, lat

# =============================================================================
# 4) ACTUADORES  -- tabla de fases; una sola fila = constante todo el tiempo
#      rc    [m]    masa móvil (+ = adelante -> nariz abajo; satura en ±0.10)
#      gamma [deg]  masa rotatoria (+ = roll a la izquierda)
#      dB    [N]    VBD (+ = sube; satura en ±4.93 con rho = 1000)
# =============================================================================
FASES = [
    # t_inicio   rc      gamma    dB      nombre
    (0.0,        0.03,    0.0,   -3.0,   "bajada"),
]
ACTUADORES_INICIO = "en_posicion"    # "en_posicion": arrancan ya en la 1a fila
#                                      "neutro": arrancan en 0 y se mueven
#                                      con su dinámica (rate limit, FT)

# =============================================================================
# 5) SALIDA
# =============================================================================
FIGURAS = [            # comentar las que no quieras
    "resumen",          # profundidad, actitud, vista superior, V, dz/dt
    "actuadores",       # posición vs comando, flotabilidad neta
    "velocidades",      # u v w p q r, alpha, beta
    "fuerzas_flujo",    # D, SF, L, T1, T2, T3 (= fcn_damping)
    "balance_pitch",    # términos del momento de pitch y su suma
    "balance_roll",
    "balance_yaw",
    "trayectoria_3d",
]
COMPARAR_CON_ANTERIOR = True   # superponer con la corrida anterior
ALFA_VALIDO_DEG = 15.0         # aviso si |alpha| pasa de esto (modelo alfa-beta)
MOSTRAR = True                 # abrir las figuras al final (zoom, pan)

# =============================================================================
# 6) CORRER
# =============================================================================
NOMBRE = "02_simulacion"
OUT = os.path.join(RAIZ, "out", NOMBRE)
ANT = os.path.join(OUT, "anterior")
ARCHIVOS = ("telemetria.npz", "entradas.json", "config_usada.toml")

entradas = dict(CONFIG=os.path.relpath(CONFIG, RAIZ), MODELO=MODELO,
                SOLVER=SOLVER, DT=DT, T_FINAL=T_FINAL, MUESTREO=MUESTREO,
                POSICION=POSICION, ACTITUD_DEG=ACTITUD_DEG,
                VELOCIDAD_MARCO=VELOCIDAD_MARCO, VELOCIDAD=VELOCIDAD,
                VEL_ANGULAR_DEG=VEL_ANGULAR_DEG, FASES=FASES,
                ACTUADORES_INICIO=ACTUADORES_INICIO)
entradas = json.loads(json.dumps(entradas))        # tuplas -> listas

# ---- revisar entradas --------------------------------------------------------
if SOLVER not in SOLVERS:
    sys.exit(f"SOLVER '{SOLVER}' no existe; opciones: {list(SOLVERS)}")
decimate = int(round(MUESTREO / DT))
if decimate < 1 or abs(decimate * DT - MUESTREO) > 1e-9:
    sys.exit(f"MUESTREO ({MUESTREO}) debe ser múltiplo de DT ({DT})")
n = int(round(T_FINAL / MUESTREO))
if ACTUADORES_INICIO not in ("en_posicion", "neutro"):
    sys.exit("ACTUADORES_INICIO debe ser 'en_posicion' o 'neutro'")
marco = {"cuerpo": "body", "inercial": "ned"}.get(VELOCIDAD_MARCO)
if marco is None:
    sys.exit("VELOCIDAD_MARCO debe ser 'cuerpo' o 'inercial'")

# ---- modelo, controlador, estado inicial ------------------------------------
p, cfg = load_config(CONFIG, MODELO, dt=DT)
tabla = [(t0, rc, np.deg2rad(gm), dB) for t0, rc, gm, dB, _ in FASES]
cs0, ctrl = make_open_loop_controller(tabla)
ecef0 = None
if cfg.use_geodesy:
    from gliderjax.geodesy import lla2ecef, R_n_e
    lon, lat = np.deg2rad(ORIGEN_GEO)
    ecef0 = (lla2ecef(np.array([lon, lat, 0.0]))
             + R_n_e(lon, lat) @ np.asarray(POSICION, dtype=float))
x0 = build_initial_state(
    p, pos_ned=POSICION, euler=np.deg2rad(ACTITUD_DEG), vel=VELOCIDAD,
    vel_frame=marco, omega=np.deg2rad(VEL_ANGULAR_DEG),
    actuators=tabla[0][1:] if ACTUADORES_INICIO == "en_posicion" else None,
    ecef0=ecef0)

m_tot = float(p.m_static + p.m_lin + p.m_rot + p.m_pist)
B0 = float(p.rho) * G0 * float(p.V_displaced)
print(f"modelo: {entradas['CONFIG']}, perfil '{MODELO}'")
print(f"solver: {SOLVER}, dt = {DT} s, {T_FINAL:.0f} s "
      f"({int(round(T_FINAL / DT))} pasos, {n} muestras cada {MUESTREO} s)")
print(f"flotabilidad neta con dB = 0: {B0 - m_tot * G0:+.3f} N "
      f"(masas {m_tot:.3f} kg, V_displaced {float(p.V_displaced):.6f} m^3)")

# ---- simular ----------------------------------------------------------------
sim = make_simulator(ctrl, cs0, cfg, dt=DT, decimate=decimate, ctrl_every=1,
                     solver=SOLVER)
t_ini = time.time()
tel = {k: np.asarray(v) for k, v in sim(p, x0, n).items()}
print(f"simulación: {time.time() - t_ini:.1f} s (incluye compilación)")

# ---- guardar; la corrida anterior pasa a anterior/ --------------------------
os.makedirs(OUT, exist_ok=True)
if os.path.exists(os.path.join(OUT, ARCHIVOS[0])):
    os.makedirs(ANT, exist_ok=True)
    for f in ARCHIVOS:
        if os.path.exists(os.path.join(OUT, f)):
            shutil.move(os.path.join(OUT, f), os.path.join(ANT, f))
scope.save_tel(tel, os.path.join(OUT, "telemetria.npz"))
with open(os.path.join(OUT, "entradas.json"), "w", encoding="utf-8") as f:
    json.dump(entradas, f, indent=1, ensure_ascii=False)
shutil.copy(CONFIG, os.path.join(OUT, "config_usada.toml"))

# ---- qué cambió respecto a la corrida anterior ------------------------------
tel_ant = None
if os.path.exists(os.path.join(ANT, "telemetria.npz")):
    print("\ncambios respecto a la corrida anterior:")
    with open(os.path.join(ANT, "entradas.json"), encoding="utf-8") as f:
        ent_ant = json.load(f)
    cambios = [f"  {k}: {ent_ant.get(k)} -> {v}"
               for k, v in entradas.items() if ent_ant.get(k) != v]
    par_ant, mod_ant = read_config(os.path.join(ANT, "config_usada.toml"))
    par, mod = read_config(CONFIG)
    cambios += [f"  {k}: {par_ant.get(k)} -> {v}  (config)"
                for k, v in par.items() if par_ant.get(k) != v]
    sw_ant = mod_ant.get(ent_ant.get("MODELO"), {})
    cambios += [f"  {k}: {sw_ant.get(k)} -> {v}  (switch)"
                for k, v in mod[MODELO].items() if sw_ant.get(k) != v]
    print("\n".join(cambios) if cambios else "  ninguno")
    if COMPARAR_CON_ANTERIOR:
        tel_ant = scope.load_tel(os.path.join(ANT, "telemetria.npz"))

# ---- avisos -----------------------------------------------------------------
print()
t = tel["t"]
if np.isnan(tel["eta"]).any():
    i = np.argmax(np.isnan(tel["eta"]).any(axis=1))
    print(f"ATENCIÓN: NaN desde t = {t[i]:.1f} s")
a_deg = np.abs(np.rad2deg(tel["alpha"]))
fuera = a_deg > ALFA_VALIDO_DEG
if fuera.any():
    print(f"ATENCIÓN: |alpha| > {ALFA_VALIDO_DEG:.0f}° el {100 * fuera.mean():.1f} % "
          f"del tiempo (desde t = {t[np.argmax(fuera)]:.1f} s, máx "
          f"{a_deg.max():.0f}°): fuera del rango del modelo alfa-beta")
z = tel["eta"][:, 2]
if cfg.surface_model == "none" and z.min() < 1.0:
    print(f"ATENCIÓN: llegó a z = {z.min():.2f} m sin modelo de superficie")
rc_cmd = np.array([f[1] for f in FASES])
if np.any(np.abs(rc_cmd) > float(p.dx_lin_max)):
    print(f"aviso: algún rc pasa de ±{float(p.dx_lin_max):.3f} m (se satura)")
dB_max = float(p.rho) * G0 * float(p.vbd_area) * float(p.vbd_max_disp)
if np.any(np.abs([f[3] for f in FASES]) > dB_max):
    print(f"aviso: algún dB pasa de ±{dB_max:.2f} N (el pistón se satura)")

# ---- resumen ----------------------------------------------------------------
fases = [(f[0], f[4]) for f in FASES]
scope.phase_summary(tel, fases, t_end=T_FINAL)
print(f"\nfinal (t = {t[-1]:.0f} s): z = {z[-1]:.1f} m, "
      f"pitch = {np.rad2deg(tel['eta'][-1, 4]):+.1f}°, "
      f"roll = {np.rad2deg(tel['eta'][-1, 3]):+.1f}°, V = {tel['V'][-1]:.3f} m/s")

# =============================================================================
# 7) GRÁFICAS
# =============================================================================
scope.KEEP_OPEN = MOSTRAR
fig = lambda nombre: os.path.join(OUT, nombre)
graficas = {
    "resumen": lambda: scope.overview(
        tel, title="Resumen", save=fig("01_resumen.png"), phases=fases),
    "actuadores": lambda: scope.actuators_plot(
        tel, title="Actuadores: posición vs comando",
        save=fig("02_actuadores.png"), phases=fases),
    "velocidades": lambda: scope.scope(
        tel, [f"nu_r:{i}" for i in range(6)] + ["alpha_deg", "beta_deg"],
        title="Velocidades en ejes cuerpo y ángulos de flujo",
        save=fig("03_velocidades.png"), phases=fases),
    "fuerzas_flujo": lambda: scope.flow_frame_wrench(
        tel, save=fig("04_fuerzas_flujo.png"), phases=fases),
    "balance_pitch": lambda: scope.moments_breakdown(
        tel, axis=4, save=fig("05_balance_pitch.png"), phases=fases),
    "balance_roll": lambda: scope.moments_breakdown(
        tel, axis=3, save=fig("06_balance_roll.png"), phases=fases),
    "balance_yaw": lambda: scope.moments_breakdown(
        tel, axis=5, save=fig("07_balance_yaw.png"), phases=fases),
    "trayectoria_3d": lambda: scope.trajectory_3d(
        tel, save=fig("08_trayectoria_3d.png")),
}
for nombre in FIGURAS:
    if nombre not in graficas:
        print(f"aviso: figura '{nombre}' no existe; opciones: {list(graficas)}")
        continue
    graficas[nombre]()
if tel_ant is not None:
    scope.compare([tel, tel_ant],
                  ["depth", "pitch_deg", "roll_deg", "yaw_deg", "V",
                   "alpha_deg", "B_net", "actuators:0"],
                  labels=["esta corrida", "anterior"],
                  title="Esta corrida vs la anterior",
                  save=fig("00_vs_anterior.png"), phases=fases)

print(f"\nresultados en {OUT}")
if MOSTRAR:
    plt.show()
