# Control y misión por waypoints

Última actualización: 2026-10-24

## 1. Arquitectura

```
FSM ─ pitch_ref ─► limitador 0.1 rad/s ─► 1/(5s+1) ─► PID pitch ─► x_g deseado ─► rc (masa móvil)
    ─ yaw_ref ───► limitador 0.03 rad/s ─► 1/(5s+1) ─► P yaw ─► roll_ref ─► P roll ─► γ (masa rotatoria)
    ─ Buoy_ref ────────────────────────────────────────────────────────────► dB ±3 N (sin lazo)
```

- FSM: `navigation.py::build_fsm` (port "literal" de tu V2, según el port).
- Moldeado de referencias: `sim.py::make_ref_pipeline`. Valores de Simulink:
  pitch ±0.1 rad/s → 1/(5s+1); yaw ±0.03 rad/s → 1/(5s+1); roll sin moldeado.
- Controladores: `autopilot.py::attitude_step`.
- Todo junto: `autopilot.py::make_waypoint_controller`.
- Sin FSM ni controladores (lazo abierto, tabla de fases):
  `autopilot.py::make_open_loop_controller`, usado en
  `escenarios/01_validacion_dinamica.py` (ver [VALIDACION.md](VALIDACION.md))
  y en `02_simulacion.py` con `MODO = "directo"`.
- Controladores sin FSM (tabla de referencias de pitch, rumbo, dB y signo de giro):
  `autopilot.py::make_attitude_controller`, usado en `02_simulacion.py` con
  `MODO = "control"` (ver [SIMULACION.md](SIMULACION.md)). El moldeado de
  referencias arranca en la actitud inicial (`init_refs`), para que no haya
  un giro falso si el rumbo inicial no es 0.
- Hay **dos prefiltros distintos**: los de los actuadores (`[actuadores]`
  del archivo de configuración, verificados) y el moldeado de las
  referencias de control (`sim.py::make_ref_pipeline`, valores en el
  código).
- En `demos/demo_mission.py` el controlador y la FSM corren cada
  `CTRL_EVERY × DT = 5 × 0.05 = 0.25 s` (el comentario del demo dice 0.5 s;
  es incorrecto).

## 2. Controladores de actitud: PROVISIONALES

**No son tus bloques `Pitch_control`, `Yaw_control` ni `Roll control` de
Simulink.** La sesión anterior los puso para que la misión pudiera volar
mientras llegaban los tuyos. Ganancias: constantes al inicio de
`autopilot.py` (líneas 32-37).

**Pitch: PID** (`KP_P = 0.015`, `KD_P = 0.08`, `KI_P = 0.003`)

```
e      = θ − θ_ref
x_g    = (1/cos θ) · (KP·e + KD·q + KI·∫e − z_g0·sin θ)
rc_cmd = m_tot · x_g / m_lin
```

- La salida es la posición deseada del centro de gravedad, convertida a
  posición de la masa móvil.
- `−z_g0·sin θ`: compensación de gravedad (z_g0 se calcula de los parámetros).
- La parte D usa la q medida, no la derivada del error.
- Anti-windup: el integrador se congela solo si la masa está saturada **y**
  el error empuja hacia más saturación.

**Yaw: P** (`K_YAW = 1.2`)

```
roll_ref = clip(K_YAW · wrap(ψ_ref − ψ), ±ROLL_MAX) · turn_sign       ROLL_MAX = 0.45 rad (~26°)
```

`turn_sign` (signo de giro; antes se llamaba `cg`) viene de la FSM: +1 en
DIVE, −1 en CLIMB, 0 en los demás estados. Invierte el sentido del giro
entre bajada y subida (ver [MODELO.md](MODELO.md), sección 6). Con
`turn_sign = 0` no hay control de rumbo.

**Roll: P** (`K_ROLL = 2.0`)

```
γ_cmd = −K_ROLL · (roll_ref − φ)          (signo −: γ > 0 produce φ < 0)
```

**Flotabilidad: sin lazo.** `dB = BUOY_SIGN · Buoy_ref`, con
`BUOY_SIGN = −1` (**marcado `CONFIRM`**). Convierte la convención de V2
(+3 bajar, −3 subir) a la física del VBD (dB negativo = menos flotación =
baja). El VBD satura en ±4.93 N (carrera completa).

## 3. FSM de waypoints (V2)

Constantes en `navigation.py` (líneas 20-36):

| Constante | Valor | Significado |
|---|---|---|
| `ACCEPT_R` | 90 m | radio de llegada al waypoint |
| `SURFACE_WAIT` | 30 s | espera en superficie (FIX) |
| `BOTTOM_WAIT` | 10 s | espera en el fondo |
| `STAB_TIME` | 10 s | estabilización al salir (SURF) |
| `PITCH_ANT` | −1.1 rad (−63°) | postura de antena |
| `PITCH_DN` / `PITCH_UP` | ∓0.6109 rad (∓35°) | pitch de bajada / subida |
| `BUOY_DN` / `BUOY_UP` | +3 / −3 | flotabilidad (convención V2) |
| `MAX_DEPTH` | 200 m | profundidad máxima planeada |
| `TURN_DEPTH` | 2 m | profundidad para pasar de SUB a DIVE |
| `DEPTH_EPS` | 4 m | margen para llegar a la profundidad planeada |
| `SURF_EPS` | 1 m | profundidad que cuenta como superficie |
| `FLARE_DEPTH` | 1 m | inicio del flare (ver problema 2) |
| `MIN_EST_DEPTH` | 20 m | profundidad mínima para estimar deriva |
| `K_GLIDE` | 1.4 | razón de planeo supuesta |
| `ALPHA_F` | 0.8 | filtro del estimador de deriva |

| # | Estado | pitch_ref | yaw_ref | Buoy (V2) | signo de giro | Sale cuando |
|---|---|---|---|---|---|---|
| 1 | FIX | −63° | planeado | −3 (positiva) | 0 | pasan 30 s → SUB, o DONE si no quedan waypoints |
| 2 | SUB | −35° | actual | +3 (negativa) | 0 | prof. > 2 m → DIVE |
| 3 | DIVE | −35° | planeado | +3 (negativa) | **+1** | llega a prof. planeada − 4 m → BOT |
| 4 | BOT | 0° | planeado | 0 (neutra) | 0 | pasan 10 s → CLIMB |
| 5 | CLIMB | +35° | planeado | −3 (positiva) | **−1** | prof. ≤ 1 m → SURF |
| 6 | DONE | −63° | actual | **0 (neutra)** | 0 | nunca (final) |
| 7 | SURF | 0° | planeado | −3 (positiva) | 0 | pasan 10 s → FIX |

En FIX (al cumplirse los 30 s): toma fix GPS, estima la deriva (si la
inmersión pasó de 20 m), revisa si llegó al waypoint, y planea rumbo y
profundidad: `prof = clip(dist_corregida / (2·K_GLIDE), 4 m, 200 m)`.

Misiones predefinidas en `navigation.py`: `MISSION_TRIANGLE`,
`MISSION_SQUARE7`, `MISSION_SQUARE_3KM`, `MISSION_LAWNMOWER`
(formato `[lon, lat, alt]`).

**Ejemplo, misión triángulo (2026-10-24, modelo completo):**
FIX 1 s → SUB 31 s → DIVE 50 s → BOT 474 s (135.6 m) → CLIMB 484 s →
SURF 922 s → FIX 932 s → … → DONE 1854 s.

## 4. Problemas conocidos

1. **DONE con flotabilidad neutra.** Pide Buoy = 0 y pitch −63°. El
   planeador queda colgado nariz abajo y se hunde despacio (4.2 → 5.6 m
   entre t = 4500 y 6000 s). **Debería tener flotabilidad positiva**
   (autor, 2026-10-24). Arreglo: `BUOY_UP` en lugar de `0.0` en `s_done`
   (`navigation.py:171`). Pendiente: confirmar si tu V2 también pide 0 (sería
   un error de V2 que en Simulink no se veía porque la simulación terminaba)
   o si es un error del port.
2. **El flare no tiene efecto.** `FLARE_DEPTH = SURF_EPS = 1 m`, así que
   en CLIMB el denominador del flare es 0: el pitch se queda en +35° hasta
   pasar a SURF. (Si la profundidad fuera exactamente 1.0 m daría 0/0 = NaN;
   en la práctica no ocurre.) Pendiente: revisar los valores en tu V2.
3. **`BUOY_SIGN` marcado `CONFIRM`**: confirmar contra el cableado entre la
   FSM y el bloque de flotabilidad en Simulink.
4. **Controladores provisionales** (sección 2). Pendiente: reemplazarlos por
   tus bloques de Simulink.
5. **Ruido GPS no implementado** (`gps_sigma_m` existe pero es TODO).
6. **Peculiaridad de V2:** en el tick en que se acaban los waypoints, la
   salida de FIX es `[0, yaw, 0, 0, 0]` en lugar de la normal (el port dice
   que es así en V2).
7. **Pitch de cada inmersión.** Hoy es fijo para toda la misión
   (`PITCH_DN`, `PITCH_UP` = ∓35°). Tu `fcn_VBD_actuator` tenía además un
   pitch automático `p_ref = ±con` según el signo de dB, que el port no
   tiene. **Decisión (autor, 2026-10-24):** quitar el pitch automático del
   actuador y dar siempre el pitch explícitamente para cada inmersión.
   Pendiente para el script de waypoints: pitch de bajada y de subida por
   inmersión (o por waypoint) como dato de la misión.
