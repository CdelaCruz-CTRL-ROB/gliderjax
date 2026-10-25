"""03_mision.py -- misión por waypoints con la máquina de estados (FSM).

Flujo de trabajo igual que 02_simulacion.py: cambias valores aquí (o en
config/cidesi.toml), corres, ves las gráficas, cambias, vuelves a correr.
Imprime qué cambió respecto a la corrida anterior.

Correr desde GLIDER_JAX\\sim\\gliderjax:

    ..\\Scripts\\python.exe escenarios\\03_mision.py

Resultados en out/03_mision/ (la corrida anterior en anterior/).
Documentación: docs/MISION.md
"""
import dataclasses
import json
import os
import shutil
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)          # para importar la librería gliderjax

import numpy as np
import matplotlib.pyplot as plt
from gliderjax import make_resumable_simulator, build_initial_state, scope, SOLVERS, fsm
from gliderjax.params import load_config, read_config
from gliderjax.geodesy import lla2ecef, R_n_e
from gliderjax.autopilot import make_waypoint_controller, waypoint_signals
from gliderjax.fsm.dive_models import DIVE_MODELS

# =============================================================================
# 1) MODELO  (parámetros y switches: editar config/cidesi.toml)
# =============================================================================
CONFIG = os.path.join(RAIZ, "config", "cidesi.toml")
MODELO = "simulink"   # perfil [modelo.<nombre>]: "simulink" | "completo"
#                       (la geodesia se enciende sola: la misión necesita lat/lon)

# =============================================================================
# 2) SOLVER
# =============================================================================
SOLVER = "ode4"       # "ode4" Runge-Kutta 4 | "ode2" Heun | "ode1" Euler
DT = 0.005            # [s] paso de integración
MUESTREO = 1.0        # [s] cada cuánto se guarda una muestra (múltiplo de DT)

# =============================================================================
# 3) DURACIÓN
# =============================================================================
DURACION = "mision"   # "mision": corre hasta que la FSM termina (DONE) y
#                                 T_DESPUES_DONE más; si no termina antes de
#                                 T_MAX se detiene y avisa TIMEOUT
#                       "tiempo": corre T_FINAL y se corta, terminen o no
#                                 los waypoints
T_MAX = 60000.0       # [s] timeout del modo "mision"
T_DESPUES_DONE = 20.0  # [s] (en Simulink, el bloque stop corta 20 s después)
T_FINAL = 3000.0      # [s] duración del modo "tiempo"

# =============================================================================
# 4) MISIÓN
# =============================================================================
# Una fila por waypoint: (lon, lat) o (lon, lat, pitch_bajada, pitch_subida).
#   lon, lat [deg]: en ese orden (Fossen / tu Simulink). OJO: Google Maps da
#   (lat, lon); si se invierten, en el Golfo "lat = -95" y el script avisa.
#   pitch [deg], opcional: el de las inmersiones HACIA ese waypoint (bajada
#   < 0, subida > 0); sin él, el de la FSM (-35 / +35).
#   La primera fila es SIEMPRE el origen (WP0): origen del marco NED y
#   punto de arranque. Las trayectorias se eligen comentando / descomentando.
WAYPOINTS = [
    # lon [deg]     lat [deg]    pitch bajada  pitch subida [deg]
    # --- origen: SIEMPRE la primera fila ---
    (-95.72275,    19.37214),     # WP0: origen (19°22'19.7"N, 95°43'21.9"W)

    # --- triángulo (el de Simulink) ---
 #   (-95.72011,    19.37464),     # WP1 (19°22'28.7"N, 95°43'12.4"W)
 #   (-95.72275,    19.37714),     # WP2 (19°22'37.7"N, 95°43'21.9"W)
    # --- ... y para el cuadro de 7 puntos, además: ---
    # (-95.71798,    19.37714),   # WP3 (~500 m al este del WP2)
    # (-95.71798,    19.37264),   # WP4 (~500 m al sur del WP3)
    # (-95.72275,    19.37264),   # WP5 (~500 m al oeste del WP4)
    # (-95.72275,    19.37714),   # WP6 (~500 m al norte del WP5)

    # --- un solo punto al norte ---
    # (-95.72275,    19.37885),

    # --- cuadro de 3 km ---
     (-95.72275,    19.40014),   # WP A: 3 km al norte del origen
     (-95.69412,    19.40014),   # WP B: 3 km al este del WP A
     (-95.69412,    19.37214),   # WP C: 3 km al sur del WP B
     (-95.70844,    19.38614),   # WP D: centro del cuadro (fin)

    # --- barrido (lawnmower) ---
    # (-95.72275,    19.39014),   # WP A: 2 km al norte
    # (-95.71321,    19.39014),   # WP B: 1 km al este
    # (-95.71321,    19.37214),   # WP C: 2 km al sur
    # (-95.70367,    19.37214),   # WP D: 1 km al este
    # (-95.70367,    19.39014),   # WP E: 2 km al norte
    # (-95.69412,    19.39014),   # WP F: 1 km al este (fin)
]
FSM = "simulink"            # máquina de estados (gliderjax/fsm/)
MODELO_INMERSION = "recta"  # gliderjax/fsm/dive_models.py
FSM_AJUSTES = {             # constantes de la FSM a cambiar (vacío = las de
    # "acceptance_radius_m": 90.0,          # Simulink; lista en
    # "max_dive_depth_m": 200.0,            # docs/CONTROL_Y_MISION.md, 3.2)
}
PERIODO_CONTROL = 0.01      # [s] FSM y controladores (múltiplo de DT)

# =============================================================================
# 5) CONDICIONES INICIALES
# =============================================================================
POSICION = (0.0, 0.0, 0.0)       # [m] N, E, z (profundidad) respecto al WP1
ACTITUD_DEG = (0.0, 0.0, 0.0)    # [deg] roll, pitch, yaw
#                                  (velocidad 0, actuadores en neutro)

# =============================================================================
# 6) SALIDA
# =============================================================================
FIGURAS = [            # comentar las que no quieras
    "mapa",             # vista superior: trayectoria, waypoints, radio de llegada
    "perfil",           # profundidad, estado de la FSM, waypoint objetivo
    "actitud",          # pitch, rumbo, roll contra sus referencias; flotabilidad
    "deriva",           # estimación de deriva y distancia al waypoint en cada fix
   # "resumen",          # el de 02_simulacion
   # "actuadores",
   # "trayectoria_3d",
]
ALFA_VALIDO_DEG = 15.0         # aviso si |alpha| pasa de esto bajo el agua
MOSTRAR = True                 # abrir las figuras al final (zoom, pan)

# =============================================================================
# 7) CORRER
# =============================================================================
NOMBRE = "03_mision"
OUT = os.path.join(RAIZ, "out", NOMBRE)
ANT = os.path.join(OUT, "anterior")
ARCHIVOS = ("telemetria.npz", "entradas.json", "config_usada.toml", "resultado.json")
BLOQUE = 500.0        # [s] simula por bloques; entre bloques revisa si terminó

entradas = dict(CONFIG=os.path.relpath(CONFIG, RAIZ), MODELO=MODELO,
                SOLVER=SOLVER, DT=DT, MUESTREO=MUESTREO, DURACION=DURACION,
                T_MAX=T_MAX, T_DESPUES_DONE=T_DESPUES_DONE, T_FINAL=T_FINAL,
                WAYPOINTS=WAYPOINTS, FSM=FSM, MODELO_INMERSION=MODELO_INMERSION,
                FSM_AJUSTES=FSM_AJUSTES, PERIODO_CONTROL=PERIODO_CONTROL,
                POSICION=POSICION, ACTITUD_DEG=ACTITUD_DEG)
entradas = json.loads(json.dumps(entradas))        # tuplas -> listas

# ---- revisar entradas --------------------------------------------------------
if SOLVER not in SOLVERS:
    sys.exit(f"SOLVER '{SOLVER}' no existe; opciones: {list(SOLVERS)}")
decimate = int(round(MUESTREO / DT))
if decimate < 1 or abs(decimate * DT - MUESTREO) > 1e-9:
    sys.exit(f"MUESTREO ({MUESTREO}) debe ser múltiplo de DT ({DT})")
ctrl_every = int(round(PERIODO_CONTROL / DT))
if ctrl_every < 1 or abs(ctrl_every * DT - PERIODO_CONTROL) > 1e-9:
    sys.exit(f"PERIODO_CONTROL ({PERIODO_CONTROL}) debe ser múltiplo de DT ({DT})")
if DURACION not in ("mision", "tiempo"):
    sys.exit("DURACION debe ser 'mision' o 'tiempo'")
if FSM not in fsm.FSMS:
    sys.exit(f"FSM '{FSM}' no existe; opciones: {list(fsm.FSMS)}")
if MODELO_INMERSION not in DIVE_MODELS:
    sys.exit(f"MODELO_INMERSION '{MODELO_INMERSION}' no existe; opciones: {list(DIVE_MODELS)}")
filas = [tuple(w) for w in WAYPOINTS]
if len(filas) < 2 or any(len(w) not in (2, 4) for w in filas):
    sys.exit("WAYPOINTS: al menos 2 filas; cada una (lon, lat) o "
             "(lon, lat, pitch_bajada, pitch_subida)")

# ---- misión, FSM, modelo, estado inicial ------------------------------------
mision = [(w[0], w[1], 0.0) for w in filas]
pitch_tramo = [(np.deg2rad(w[2]), np.deg2rad(w[3])) if len(w) == 4
               else (np.nan, np.nan) for w in filas]
try:
    nav = fsm.make(FSM, mision, dive_model=DIVE_MODELS[MODELO_INMERSION],
                   leg_pitch=pitch_tramo, **FSM_AJUSTES)
except (ValueError, TypeError) as e:
    sys.exit(f"misión / FSM_AJUSTES: {e}")
n_wp = len(nav.waypoints_ned)
n_obj = n_wp - 1          # waypoints objetivo (sin el origen, WP0)
# la FSM numera desde 1 (WP0 = índice 1); aquí se muestra como en tu lista

p, cfg = load_config(CONFIG, MODELO, dt=DT)
geo_forzada = not cfg.use_geodesy
cfg = dataclasses.replace(cfg, use_geodesy=True)
lon0, lat0 = np.deg2rad(nav.origin_lla_deg[0:2])
ecef0 = (np.asarray(lla2ecef(np.array([lon0, lat0, 0.0])))
         + np.asarray(R_n_e(lon0, lat0)) @ np.asarray(POSICION, dtype=float))
x0 = build_initial_state(p, pos_ned=POSICION, euler=np.deg2rad(ACTITUD_DEG),
                         ecef0=ecef0)
roll0, pitch0, yaw0 = np.deg2rad(ACTITUD_DEG)
cs0, ctrl = make_waypoint_controller(nav, PERIODO_CONTROL,
                                     init_refs=(pitch0, yaw0, roll0))

print(f"modelo: {entradas['CONFIG']}, perfil '{MODELO}'"
      + (" (+ geodesia, la necesita la misión)" if geo_forzada else ""))
print(f"solver: {SOLVER}, dt = {DT} s, muestra cada {MUESTREO} s; "
      f"FSM y control cada {PERIODO_CONTROL} s")
print(f"FSM: {nav.name}")
print("duración: " + (f"hasta terminar la misión + {T_DESPUES_DONE:.0f} s, "
                      f"timeout en {T_MAX:.0f} s" if DURACION == "mision"
                      else f"{T_FINAL:.0f} s fijos"))
cfg_fsm = nav.config
print("\nWP   lon [deg]   lat [deg]    N [m]    E [m]  tramo [m]  pitch bajada/subida")
for i, (w, ne) in enumerate(zip(filas, nav.waypoints_ned)):
    tramo = "" if i == 0 else f"{np.hypot(*(ne - nav.waypoints_ned[i - 1])):9.1f}"
    if i == 0:
        pt = "(origen)"
    elif len(w) == 4:
        pt = f"{w[2]:+.1f} / {w[3]:+.1f}"
    else:
        pt = (f"{np.rad2deg(cfg_fsm.pitch_down_rad):+.1f} / "
              f"{np.rad2deg(cfg_fsm.pitch_up_rad):+.1f} (FSM)"
              if hasattr(cfg_fsm, "pitch_down_rad") else "(FSM)")
    print(f"{i:<3d} {w[0]:10.5f} {w[1]:10.5f} {ne[0]:8.1f} {ne[1]:8.1f} "
          f"{tramo:>10s}  {pt}")

# ---- simular por bloques ----------------------------------------------------
start, run = make_resumable_simulator(ctrl, cs0, cfg, dt=DT, decimate=decimate,
                                      ctrl_every=ctrl_every, solver=SOLVER)
n_bloque = int(round(BLOQUE / MUESTREO))
t_tope = T_MAX if DURACION == "mision" else T_FINAL
t_fin, t_done, hay_nan = t_tope, None, False
partes = []
carry = start(p, x0)
t_ini = time.time()
print()
while True:
    carry, tb = run(p, carry, n_bloque)
    tb = {k: np.asarray(v) for k, v in tb.items()}
    partes.append(tb)
    sb = waypoint_signals(tb, nav)
    hecho = sb["mission_complete"] > 0.5
    if t_done is None and hecho.any():
        t_done = float(tb["t"][np.argmax(hecho)])
        if DURACION == "mision":
            t_fin = min(t_done + T_DESPUES_DONE, t_tope)
    hay_nan = bool(np.isnan(tb["eta"]).any())
    k = int(sb["current_wp_index"][-1])
    print(f"  t = {tb['t'][-1]:7.0f} s   {nav.state_names[int(sb['glider_state'][-1]) - 1]:12s} "
          + (f"hacia WP{k - 1}/{n_obj}" if k <= n_wp else "waypoints terminados")
          + f"   z = {tb['eta'][-1, 2]:6.1f} m   ({time.time() - t_ini:.0f} s)")
    if tb["t"][-1] >= t_fin - 1e-6 or hay_nan:
        break
tel = {k: np.concatenate([q[k] for q in partes]) for k in partes[0]}
keep = tel["t"] <= t_fin + 1e-6
tel = {k: v[keep] for k, v in tel.items()}
t = tel["t"]
sig = waypoint_signals(tel, nav)
alcanzados = int(np.clip(sig["current_wp_index"].max() - 2, 0, n_obj))

# ---- resultado --------------------------------------------------------------
if hay_nan:
    resultado = "nan"
    msg = f"ATENCIÓN: NaN desde t = {t[np.argmax(np.isnan(tel['eta']).any(axis=1))]:.0f} s"
elif t_done is not None:
    resultado = "completa"
    msg = f"MISIÓN COMPLETA: DONE en t = {t_done:.0f} s ({alcanzados}/{n_obj} waypoints)"
elif DURACION == "mision":
    resultado = "timeout"
    msg = (f"TIMEOUT: la misión no terminó en T_MAX = {T_MAX:.0f} s; "
           f"llegó a {alcanzados} de {n_obj} waypoints")
else:
    resultado = "tiempo"
    msg = (f"tiempo cumplido (T_FINAL = {T_FINAL:.0f} s) sin terminar: "
           f"llegó a {alcanzados} de {n_obj} waypoints")
print(f"\n{msg}")
print(f"simulación: {t[-1]:.0f} s en {time.time() - t_ini:.1f} s (incluye compilación)")

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
with open(os.path.join(OUT, "resultado.json"), "w", encoding="utf-8") as f:
    json.dump(dict(resultado=resultado, mensaje=msg, t_done=t_done,
                   waypoints_alcanzados=alcanzados, waypoints=n_obj,
                   t_simulado=float(t[-1])), f, indent=1, ensure_ascii=False)

# ---- qué cambió respecto a la corrida anterior ------------------------------
if os.path.exists(os.path.join(ANT, "entradas.json")):
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
    if os.path.exists(os.path.join(ANT, "resultado.json")):
        with open(os.path.join(ANT, "resultado.json"), encoding="utf-8") as f:
            print(f"  (anterior: {json.load(f)['mensaje']})")

# ---- inmersiones ------------------------------------------------------------
eta, st = tel["eta"], sig["glider_state"]
entra = lambda s: np.flatnonzero((st[1:] == s) & (st[:-1] != s)) + 1
i_sub, i_sal = entra(2), entra(7)          # SUBMERGE, SURFACE_WAIT
i_inf = []                                 # punto de inflexión (el más hondo) de cada una
print("\ninmersión  inicio [s]  hacia  rumbo plan  prof plan  prof máx  duración  "
      "al salir: dist al WP")
for j, i0 in enumerate(i_sub):
    i1 = next((i for i in i_sal if i > i0), len(t) - 1)
    i_inf.append(i0 + int(np.argmax(eta[i0:i1 + 1, 2])))
    k = int(sig["current_wp_index"][i0])
    d_wp = np.hypot(*(eta[i1, 0:2] - nav.waypoints_ned[min(k, n_wp) - 1]))
    print(f"{j + 1:6d}  {t[i0]:12.0f}  WP{k - 1:<3d} "
          f"{np.rad2deg(sig['current_yaw_ref'][i0]):9.1f}°  "
          f"{sig['current_dive_depth_m'][i0]:7.1f} m  {eta[i_inf[-1], 2]:6.1f} m  "
          f"{t[i1] - t[i0]:6.0f} s  {d_wp:10.1f} m"
          + ("" if i1 in i_sal else "  (no salió)"))

# ---- avisos -----------------------------------------------------------------
abajo = eta[:, 2] > 5.0                    # en superficie alpha no significa nada
a_deg = np.abs(np.rad2deg(tel["alpha"]))
fuera = abajo & (a_deg > ALFA_VALIDO_DEG)
if fuera.any():
    print(f"\nATENCIÓN: |alpha| > {ALFA_VALIDO_DEG:.0f}° bajo el agua (z > 5 m) el "
          f"{100 * fuera.sum() / max(abajo.sum(), 1):.1f} % del tiempo sumergido "
          f"(máx {a_deg[abajo].max():.0f}°): fuera del rango del modelo alfa-beta")

# =============================================================================
# 8) GRÁFICAS
# =============================================================================
scope.KEEP_OPEN = MOSTRAR
fig = lambda nombre: os.path.join(OUT, nombre)
titulo = f"{FSM}, {MODELO_INMERSION}: {msg}"
radio = getattr(nav.config, "acceptance_radius_m", None)


def terminar(f, nombre):
    f.tight_layout()
    f.savefig(fig(nombre), dpi=120, bbox_inches="tight")
    if not MOSTRAR:
        plt.close(f)


def figura_mapa():
    """Vista superior con el formato de tus figuras de Simulink: Norte X en
    el eje horizontal, Este Y en el vertical."""
    from matplotlib.lines import Line2D
    X, Y = eta[:, 0], eta[:, 1]
    wps = nav.waypoints_ned
    r = radio or 0.0
    xs = np.concatenate([X, wps[:, 0] - r, wps[:, 0] + r])
    ys = np.concatenate([Y, wps[:, 1] - r, wps[:, 1] + r])
    m = 0.06 * max(np.ptp(xs), np.ptp(ys), 100.0)
    x0, x1, y0, y1 = xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m
    corriente = np.asarray(p.current_n, dtype=float)[0:2]   # [N, E] del TOML
    v_c = float(np.hypot(*corriente))
    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):
        f, ax = plt.subplots(figsize=(11, 8))
        manijas = [Line2D([], [], color="red", lw=2.5, label="Trayectoria")]
        if v_c > 0:                        # corriente uniforme: flechas en una malla
            gx, gy = np.meshgrid(np.linspace(x0, x1, 13), np.linspace(y0, y1, 13))
            largo = 0.05 * max(x1 - x0, y1 - y0)
            u, v = corriente / v_c * largo
            ax.quiver(gx, gy, np.full_like(gx, u), np.full_like(gy, v),
                      angles="xy", scale_units="xy", scale=1, color="#5DADE2",
                      width=0.0035, alpha=0.9, zorder=1)
            manijas.append(Line2D([], [], color="#5DADE2", lw=2.5,
                                  label=f"Corriente marina ({v_c:.2f} m/s)"))
        ax.plot(X, Y, "-", color="red", lw=2.5, zorder=3)
        nombres = {}                       # waypoints repetidos: una etiqueta
        for i, (wn, we) in enumerate(wps[1:], start=1):   # WP0 = origen = inicio
            nombres.setdefault((round(wn), round(we)), []).append(f"WP{i}")
        for (wn, we), ns in nombres.items():
            ax.plot(wn, we, "x", color="red", ms=11, mew=2.5, zorder=4)
            if radio:
                ax.add_patch(plt.Circle((wn, we), radio, fill=False, color="red",
                                        ls="--", lw=1.5, zorder=4))
            ax.annotate(", ".join(ns), (wn, we), textcoords="offset points",
                        xytext=(10, 10), fontsize=9, color="0.3")
        manijas += [Line2D([], [], color="red", marker="x", ms=9, mew=2.5,
                           ls="none", label="Target"),
                    Line2D([], [], color="red", ls="--", lw=1.5,
                           label="Radio de aceptación")]
        ax.plot(X[i_inf], Y[i_inf], "v", color="0.2", mec="black", ms=10, zorder=5)
        for i in i_inf:
            ax.annotate(f"{eta[i, 2]:.0f} m", (X[i], Y[i]), textcoords="offset points",
                        xytext=(7, -12), fontsize=8, color="0.2", zorder=6)
        ax.plot(X[i_sal], Y[i_sal], "^", color="cyan", mec="black", ms=10, zorder=5)
        ax.plot(X[0], Y[0], "*", color="#22BB22", mec="black", ms=16, zorder=6)
        manijas += [Line2D([], [], color="0.2", mec="black", marker="v", ms=9,
                           ls="none", label="Punto de inflexión"),
                    Line2D([], [], color="cyan", mec="black", marker="^", ms=9,
                           ls="none", label="Punto de superficie"),
                    Line2D([], [], color="#22BB22", mec="black", marker="*", ms=13,
                           ls="none", label="Inicio")]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("Norte $X$ (m)", fontsize=13)
        ax.set_ylabel("Este $Y$ (m)", fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(handles=manijas, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=10, framealpha=1.0)      # afuera: no tapa nada
        ax.set_title("Navegación por waypoints"
                     + (" bajo corriente" if v_c > 0 else "") + f"\n{msg}",
                     fontsize=13)
        terminar(f, "10_mapa.png")


def figura_perfil():
    f, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, eta[:, 2], label="profundidad")
    plan = np.where((st >= 2) & (st <= 4), sig["current_dive_depth_m"], np.nan)
    ax[0].plot(t, plan, "--", label="profundidad planeada")
    ax[0].invert_yaxis()
    ax[0].set_ylabel("z [m]")
    ax[1].step(t, st, where="post")
    ax[1].set_yticks(range(1, len(nav.state_names) + 1))
    ax[1].set_yticklabels(nav.state_names, fontsize=7)
    ax[1].set_ylabel("estado FSM")
    ax[2].step(t, np.minimum(sig["current_wp_index"], n_wp) - 1, where="post")
    ax[2].set_yticks(range(0, n_wp))
    ax[2].set_ylabel("WP objetivo")
    ax[2].set_xlabel("t [s]")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper right")
    f.suptitle(titulo, fontsize=9)
    terminar(f, "11_perfil.png")


def figura_actitud():
    usada = tel["ctrl_state"][:, 12:15]             # [pitch, roll, yaw]
    f, ax = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    ax[0].plot(t, np.rad2deg(eta[:, 4]), label="pitch")
    ax[0].plot(t, np.rad2deg(usada[:, 0]), label="referencia usada")
    ax[0].plot(t, np.rad2deg(sig["pitch_ref"]), "--", lw=0.8, label="FSM")
    ax[0].set_ylabel("pitch [deg]")
    ax[1].plot(t, np.rad2deg(eta[:, 5]), label="rumbo")
    ax[1].plot(t, np.rad2deg(usada[:, 2]), label="referencia usada")
    d = sig["yaw_ref"] - usada[:, 2]               # la de la FSM, cerca de la usada
    ax[1].plot(t, np.rad2deg(usada[:, 2] + np.arctan2(np.sin(d), np.cos(d))),
               "--", lw=0.8, label="FSM")
    ax[1].set_ylabel("rumbo [deg]")
    ax[2].plot(t, np.rad2deg(eta[:, 3]), label="roll")
    ax[2].plot(t, np.rad2deg(usada[:, 1]), label="referencia (del control de yaw)")
    ax[2].set_ylabel("roll [deg]")
    ax[3].plot(t, tel["cmds"][:, 2], "--", label="comando dB")
    ax[3].plot(t, tel["B_net"], label="flotabilidad neta")
    ax[3].set_ylabel("[N]")
    ax[4].step(t, sig["turn_sign"], where="post")
    ax[4].set_ylabel("signo de giro")
    ax[4].set_xlabel("t [s]")
    for a in ax:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8, loc="upper right")
    f.suptitle(titulo, fontsize=9)
    terminar(f, "12_actitud.png")


def figura_deriva():
    f, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].step(t, sig["D_norm_n"], where="post", label="D_norm N")
    ax[0].step(t, sig["D_norm_e"], where="post", label="D_norm E")
    ax[0].set_ylabel("deriva estimada\n[m por m de prof.]")
    ax[1].step(t, sig["dist_to_wp_last_fix"], where="post",
               label="distancia al WP en el último fix")
    if radio:
        ax[1].axhline(radio, color="r", ls="--", lw=0.8, label="radio de llegada")
    ax[1].set_ylabel("[m]")
    ax[1].set_xlabel("t [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")
    f.suptitle(titulo, fontsize=9)
    terminar(f, "13_deriva.png")


graficas = {
    "mapa": figura_mapa,
    "perfil": figura_perfil,
    "actitud": figura_actitud,
    "deriva": figura_deriva,
    "resumen": lambda: scope.overview(tel, title="Resumen",
                                      save=fig("01_resumen.png")),
    "actuadores": lambda: scope.actuators_plot(
        tel, title="Actuadores: posición vs comando", save=fig("02_actuadores.png")),
    "trayectoria_3d": lambda: scope.trajectory_3d(tel, save=fig("08_trayectoria_3d.png")),
}
for nombre in FIGURAS:
    if nombre not in graficas:
        print(f"aviso: figura '{nombre}' no existe; opciones: {list(graficas)}")
        continue
    graficas[nombre]()

print(f"\nresultados en {OUT}")
if MOSTRAR:
    plt.show()
