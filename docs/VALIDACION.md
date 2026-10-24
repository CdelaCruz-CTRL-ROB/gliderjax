# Validación de la dinámica en lazo abierto

Última actualización: 2026-10-24

Script: [`escenarios/01_validacion_dinamica.py`](../escenarios/01_validacion_dinamica.py)

## Qué hace

1. Usa el **modelo equivalente a tu Simulink**, con todos los switches
   escritos explícitamente: sin superficie, sin crossflow, sin
   geolocalización, sin saturación de alpha. Además usa la masa añadida
   completa de tu `Mbr+Ma fcn` (`MA_SIMULINK`) y su rotación de la inercia
   de la masa rotatoria (`rot_inertia_mode="RtIR"`); ver
   [MODELO.md](MODELO.md), sección 2.2.
2. Mueve los actuadores según una **tabla de fases** (comandos constantes
   por tramos, como un Signal Builder).
3. Imprime los **valores cuasi-estacionarios de cada fase**.
4. Guarda 9 figuras y la telemetría en `out/01_validacion_dinamica/`.
5. Opcional: se compara contra un CSV exportado de Simulink.

## Cómo correr

Desde `GLIDER_JAX\sim\gliderjax`:

```
..\Scripts\python.exe escenarios\01_validacion_dinamica.py
```

Tarda ~2 s (incluida la compilación). Con `MOSTRAR = True` las figuras se
abren al final, con zoom y pan como en MATLAB; el script termina al cerrarlas.

## Dónde cambiar cada cosa (todo está en el script)

| Quiero cambiar… | Sección del script |
|---|---|
| un parámetro físico (KM, masas, MA…) | 1, con `p = p._replace(...)` |
| un switch del modelo | 2, `ModelConfig(...)` |
| los comandos o la duración de las fases | 3, tabla `FASES` y `T_FINAL` |
| profundidad, pitch, velocidad iniciales | 4, `initial_state(...)` |
| paso de integración o resolución de las gráficas | 5, `DT`, `DECIMATE` |
| comparar con Simulink | 6, `SIMULINK_CSV` |
| que no se abran las figuras | 6, `MOSTRAR = False` |

Para una prueba distinta conviene **copiar el script** con otro nombre (por
ejemplo `01b_espiral.py`) y cambiar `NOMBRE`, así cada prueba guarda sus
resultados en su propia carpeta.

## Prueba por defecto

| Fase | Desde | rc | γ | dB |
|---|---|---|---|---|
| bajada recta | 0 s | +30 mm | 0° | −3 N |
| bajada con giro | 300 s | +30 mm | +30° | −3 N |
| subida con giro | 600 s | −30 mm | +30° | +3 N |
| subida recta | 900 s | −30 mm | 0° | +3 N |

Arranca a 30 m, pitch −0.2 rad, u = 0.2 m/s, actuadores en neutro. Recorre
de 30 a 252 m y termina a 44 m, lejos de la superficie (que no está
modelada en esta configuración; el script avisa si se acerca a menos de 2 m).

Prueba longitudinal (planeo de bajada y subida) y lateral (giro con roll e
inversión del giro) en una sola corrida.

## Resumen por fase (consola)

Promedio sobre el último 30 % de cada fase:

| Columna | Significado |
|---|---|
| pitch, alpha | ángulos [°] |
| path | ángulo de la trayectoria [°], + = sube (θ − α en vuelo recto) |
| V | rapidez relativa al agua [m/s] |
| dz/dt | velocidad vertical [m/s], + = baja |
| roll | [°], − = ala izquierda abajo |
| yaw_rate | dψ/dt [°/s], − = gira a la izquierda (vista desde arriba) |
| glide | razón de planeo = velocidad horizontal / vertical |

**Resultados con la prueba por defecto (2026-10-24):**

```
fase              pitch    alpha    path      V      dz/dt    roll     yaw_rate  glide
bajada recta     -38.464  +1.339  -39.803  +0.565  +0.362   +0.000   +0.000   1.200
bajada con giro  -39.987  +1.253  -41.747  +0.577  +0.384  -25.932   -1.204   1.121
subida con giro  +38.176  -1.342  +40.016  +0.566  -0.364  -25.737   +1.286   1.191
subida recta     +36.596  -1.431  +38.027  +0.553  -0.341   -0.000   -0.000   1.279
```

Nota: aquí yaw_rate es dψ/dt (ángulo de Euler). En la prueba de giro de
[MODELO.md](MODELO.md) sección 6 se reportó r (velocidad angular en ejes
cuerpo); por eso los números difieren (dψ/dt = (q·sin φ + r·cos φ)/cos θ).

## Figuras

| Archivo | Contenido | Qué revisar |
|---|---|---|
| `01_resumen.png` | profundidad, pitch/α/trayectoria, roll, yaw vs t; vista superior; V; dz/dt | valores estacionarios, transitorios al cambiar de fase, sentido del giro |
| `02_trayectoria_3d.png` | trayectoria 3D | forma general (espirales) |
| `03_actuadores.png` | cada actuador contra su comando; flotabilidad neta vs dB; pistón | tiempos de respuesta, limitadores de tasa, saturaciones |
| `04_velocidades_cuerpo.png` | u, v, w, p, q, r, α, β | |
| `05_fuerzas_marco_flujo.png` | −D, SF, −L, T1, T2, T3 | son **los vectores Fh y Th de tu `fcn_damping.m`**: comparables directo con Simulink |
| `06/07/08_balance_*.png` | cada término de fuerza/momento sobre roll, pitch y yaw, y su suma | la suma (= M·ν̇) debe ser ~0 en estacionario; un término fuera de lo esperado señala dónde está el error |
| `09_restauracion.png` | fuerzas y momentos de peso + flotación | |
| `10_jax_vs_simulink.png` | JAX y Simulink superpuestos (solo con CSV) | |

## Comparar con Simulink

1. En Simulink, la **misma prueba**: misma tabla de comandos, mismas
   condiciones iniciales, actuadores en neutro al inicio. Solver de paso fijo
   **ode4 con el mismo paso (0.05 s)**; si no, las diferencias del solver
   dominan la comparación.
2. Exportar a CSV las columnas `t, x, y, z, phi, theta, psi, u, v, w, p, q, r`:
   ```matlab
   M = [t, x, y, z, phi, theta, psi, u, v, w, p, q, r];
   writematrix(M, "lazo_abierto.csv");
   ```
3. Poner la ruta en `SIMULINK_CSV` (sección 6 del script) y correr.

Salida: la figura `10_jax_vs_simulink.png` y una tabla RMS por estado con
PASS/FAIL (`validation/compare_simulink.py`). Tolerancias: posición 0.5 m,
actitud 0.5°, velocidad 0.005 m/s, velocidades angulares 0.05 °/s.

## Volver a graficar sin simular

```python
import sys; sys.path.insert(0, r"C:\Users\ANON\Desktop\GLIDER_JAX\sim\gliderjax")
import matplotlib.pyplot as plt
from gliderjax import scope

tel = scope.load_tel(r"...\out\01_validacion_dinamica\telemetria.npz")
scope.scope(tel, ["pitch_deg", "alpha_deg", "V", "tau_flow_flow:2"])
plt.show()
```

`scope.compare([tel_a, tel_b], señales, labels=[...])` superpone dos
corridas, por ejemplo antes y después de cambiar un parámetro.

## Señales disponibles

Sintaxis: `"nombre"` o `"nombre:índice"`.

| Señal | Contenido |
|---|---|
| `eta` (6), `nu_r` (6), `nu` (6) | posición/actitud, velocidad relativa y absoluta en ejes cuerpo |
| `eta_dot` (6), `nu_r_dot` (6) | derivadas |
| `pos_n`, `vel_n`, `accel_n` (3) | cinemática inercial NED |
| `alpha`, `beta`, `V` | ángulos de flujo [rad] y rapidez |
| `tau_flow_body` (6), `tau_flow_flow` (6) | hidrodinámica en ejes cuerpo / marco del flujo |
| `tau_drot` (6) | amortiguamiento rotacional (D_body) |
| `tau_coriolis_rb` (6), `tau_CA` (6) | Coriolis de cuerpo rígido / de masa añadida |
| `tau_restoring` (6) | peso + flotación |
| `tau_surface` (6), `tau_crossflow` (6) | superficie y crossflow (cero si están apagados) |
| `r_g` (3), `B`, `x_b`, `B_net` | centro de masa, flotación, centro de flotación, flotabilidad neta |
| `actuators` (3), `cmds` (3) | [rc, γ, pistón] y sus comandos [rc, γ, dB] |
| derivadas | `alpha_deg`, `beta_deg`, `roll_deg`, `pitch_deg`, `yaw_deg`, `depth`, `path_deg`, `depth_rate`, `speed_h` |

Índices de los vectores de 6: 0 = X/u, 1 = Y/v, 2 = Z/w, 3 = roll/p,
4 = pitch/q, 5 = yaw/r.
