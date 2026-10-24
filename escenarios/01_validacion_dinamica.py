"""01_validacion_dinamica.py -- validación de la dinámica en LAZO ABIERTO.

Modelo equivalente a tu Simulink: sin superficie, sin crossflow, sin
geolocalización y sin saturación de alpha. Los actuadores siguen una tabla
de fases con comandos constantes por tramos (como un Signal Builder).

Correr desde GLIDER_JAX\\sim\\gliderjax:

    ..\\Scripts\\python.exe escenarios\\01_validacion_dinamica.py

Resultados en out/01_validacion_dinamica/: figuras PNG + telemetria.npz.
Documentación: docs/VALIDACION.md
"""
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)          # para importar la librería gliderjax

import numpy as np
import matplotlib.pyplot as plt
from gliderjax import ModelConfig, make_simulator, initial_state, scope
from gliderjax.params import cidesi_defaults
from gliderjax.autopilot import make_open_loop_controller

# =============================================================================
# 1) PARÁMETROS FÍSICOS (masas, coeficientes, límites)
# =============================================================================
p = cidesi_defaults()

# Para cambiar un parámetro SIN editar params.py (devuelve una copia):
# p = p._replace(KM=p.KM * 1.10)        # +10 % rigidez de pitch
# p = p._replace(disturb_xg=0.01)          # perturbación de CG de 1 cm
# Lista completa de campos: gliderjax/params.py (GliderParams).

# =============================================================================
# 2) MODELO -- todos los switches explícitos (este conjunto = tu Simulink)
#    Cambiar cualquiera recompila (~5-10 s). Ver docs/MODELO.md, sección 7.
# =============================================================================
cfg = ModelConfig(
    kmr_mode="signum",            # KMR*sign(alpha), como en Simulink
    kmy_cubic=True,               # T3 = (KMYLIN*beta + KMYCUB*beta^3) V^2
    use_rot_damping=True,         # D_body
    use_alpha_saturation=False,   # NO está en Simulink
    surface_model="simulink",     # tu fcn_buoyancy_surface (2 puntos)
    use_crossflow=False,          # agregado por el port
    use_geodesy=False,            # sin geolocalización
    use_ca_omega=False,           # Coriolis de masa añadida: apagado
    use_ca_translational=False,   #   (Simulink anula C_A)
    replicate_disturb_wiring=True,  # perturbación de CG cableada como Simulink
)

# =============================================================================
# 3) COMANDOS EN LAZO ABIERTO -- tabla de fases
#    Desde t_inicio, los comandos se mantienen hasta la siguiente fila.
#      rc    [m]    masa móvil (+ = adelante -> nariz abajo; satura en +-0.10)
#      gamma [deg]  masa rotatoria (+ = roll a la izquierda)
#      dB    [N]    flotabilidad (- = baja, + = sube; satura en +-4.93)
# =============================================================================
FASES = [
    # t_inicio   rc      gamma    dB      nombre
    (0.0,        0.03,    0.0,   -3.0,   "bajada recta"),
    (300.0,      0.03,   30.0,   -3.0,   "bajada con giro"),
    (600.0,     -0.03,   30.0,   +3.0,   "subida con giro"),
    (900.0,     -0.03,    0.0,   +3.0,   "subida recta"),
]
T_FINAL = 1200.0                  # [s]

# =============================================================================
# 4) CONDICIONES INICIALES (actuadores en neutro)
#    Sin modelo de superficie: mantener el planeador lejos de z = 0.
# =============================================================================
x0 = initial_state(depth=30.0, pitch=-0.2, u=0.2, roll=0.0)
# Cualquier otro estado: x0 = x0.at[INDICE].set(VALOR)
#   [0:3] x, y, z   [3:6] phi, theta, psi [rad]   [6:12] u v w p q r

# =============================================================================
# 5) AJUSTES DEL SIMULADOR
# =============================================================================
DT = 0.05             # paso RK4 [s]
DECIMATE = 4          # pasos por muestra guardada: 4 -> cada 0.2 s

# =============================================================================
# 6) COMPARACIÓN CON SIMULINK (opcional)
#    CSV con columnas [t, x, y, z, phi, theta, psi, u, v, w, p, q, r],
#    misma tabla de comandos y mismas condiciones iniciales.
#    Ver validation/compare_simulink.py. None = sin comparación.
# =============================================================================
SIMULINK_CSV = None   # ej.: os.path.join(RAIZ, "validation", "lazo_abierto.csv")

MOSTRAR = True        # True: abre las figuras al final (zoom, pan)

# =============================================================================
# 7) CORRER
# =============================================================================
NOMBRE = "01_validacion_dinamica"
OUT = os.path.join(RAIZ, "out", NOMBRE)

cs0, ctrl = make_open_loop_controller(
    [(t0, rc, np.deg2rad(gm), dB) for t0, rc, gm, dB, _ in FASES])
sim = make_simulator(ctrl, cs0, cfg, dt=DT, decimate=DECIMATE, ctrl_every=1)
n = int(round(T_FINAL / (DT * DECIMATE)))

t_ini = time.time()
tel = sim(p, x0, n)
tel = {k: np.asarray(v) for k, v in tel.items()}
print(f"simulación: {T_FINAL:.0f} s simulados en {time.time() - t_ini:.1f} s "
      f"(incluye compilación)")
scope.save_tel(tel, os.path.join(OUT, "telemetria.npz"))

# ---- chequeos básicos -------------------------------------------------------
if np.isnan(tel["eta"]).any():
    print("ATENCIÓN: la simulación tiene NaN")
z_min = tel["eta"][:, 2].min()
if z_min < 2.0:
    print(f"ATENCIÓN: llegó a {z_min:.1f} m de profundidad. Sin modelo de "
          f"superficie, cerca de z = 0 los resultados no son físicos.")

# ---- valores cuasi-estacionarios por fase -----------------------------------
fases = [(f[0], f[4]) for f in FASES]
print()
scope.phase_summary(tel, fases, t_end=T_FINAL)

# =============================================================================
# 8) GRÁFICAS
# =============================================================================
scope.KEEP_OPEN = MOSTRAR
fig = lambda nombre: os.path.join(OUT, nombre)

scope.overview(tel, title="Validación lazo abierto: resumen",
               save=fig("01_resumen.png"), phases=fases)
scope.trajectory_3d(tel, save=fig("02_trayectoria_3d.png"))
scope.actuators_plot(tel, title="Actuadores: posición vs comando",
                     save=fig("03_actuadores.png"), phases=fases)
scope.scope(tel, [f"nu_r:{i}" for i in range(6)] + ["alpha_deg", "beta_deg"],
            title="Velocidades en ejes cuerpo y ángulos de flujo",
            save=fig("04_velocidades_cuerpo.png"), phases=fases)
scope.flow_frame_wrench(tel, save=fig("05_fuerzas_marco_flujo.png"),
                        phases=fases)                 # = Fh, Th de fcn_damping.m
scope.moments_breakdown(tel, axis=3, save=fig("06_balance_roll.png"), phases=fases)
scope.moments_breakdown(tel, axis=4, save=fig("07_balance_pitch.png"), phases=fases)
scope.moments_breakdown(tel, axis=5, save=fig("08_balance_yaw.png"), phases=fases)
scope.scope_vector6(tel, "tau_restoring", "Restauración (peso + flotación)",
                    save=fig("09_restauracion.png"), phases=fases)

if SIMULINK_CSV:
    from validation.compare_simulink import compare
    ref = scope.from_simulink_csv(SIMULINK_CSV)
    scope.compare([tel, ref], ["depth", "pitch_deg", "roll_deg", "yaw_deg",
                               "nu_r:0", "nu_r:2", "nu_r:4", "nu_r:5"],
                  labels=["JAX", "Simulink"], title="JAX vs Simulink",
                  save=fig("10_jax_vs_simulink.png"), phases=fases)
    print()
    compare(SIMULINK_CSV, tel)

print(f"\nfiguras y telemetría en {OUT}")
if MOSTRAR:
    plt.show()
