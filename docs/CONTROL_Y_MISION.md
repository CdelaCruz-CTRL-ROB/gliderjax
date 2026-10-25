# Control y misión por waypoints

Última actualización: 2026-10-25

## 1. Arquitectura

```
FSM ─ pitch_ref ─► limitador 0.1 rad/s ─► 1/(5s+1) ─► PID pitch ─► x_g deseado ─► rc (masa móvil)
    ─ yaw_ref ───► limitador 0.03 rad/s ─► 1/(5s+1) ─► PID yaw ─► roll_ref ─► PID roll ─► u = −y_g deseado ─► γ (masa rotatoria)
    ─ Buoy_ref ────────────────────────────────────────────────────────────► dB ±3 N (sin lazo)
```

- FSM: una por archivo en `gliderjax/fsm/`. La tuya (Simulink, versión
  2026-10) es `fsm/simulink.py`; se elige por nombre:
  `fsm.make("simulink", mision)`. Lo común a todas las FSM (sensores,
  salidas, GPS, misiones) está en `navigation.py`. Sección 3.
- Moldeado de referencias: `sim.py::make_ref_pipeline`. Valores de Simulink:
  pitch ±0.1 rad/s → 1/(5s+1); yaw ±0.03 rad/s → 1/(5s+1); roll sin moldeado.
- Controladores: `autopilot.py::pitch_control`, `roll_control`,
  `yaw_control`; los conecta `attitude_step` (sección 5).
- Todo junto: `autopilot.py::make_waypoint_controller`.
- Sin FSM ni controladores (lazo abierto, tabla de fases):
  `autopilot.py::make_open_loop_controller`, usado en
  `escenarios/01_validacion_dinamica.py` (ver [VALIDACION.md](VALIDACION.md))
  y en `02_simulacion.py` con `MODO = "directo"`.
- Controladores sin FSM: `autopilot.py::make_attitude_controller`, usado
  en `02_simulacion.py` con `MODO = "pitch_roll"` (pitch y roll cada uno
  por su lado, referencias de una tabla) o `MODO = "pitch_roll_yaw"` (el
  control de yaw da la referencia de roll; signo de giro automático). Ver [SIMULACION.md](SIMULACION.md). El
  moldeado de referencias arranca en la actitud inicial (`init_refs`), para
  que no haya un giro falso si el rumbo inicial no es 0.
- Hay **dos prefiltros distintos**: los de los actuadores (`[actuadores]`
  del archivo de configuración, verificados) y el moldeado de las
  referencias de control (`sim.py::make_ref_pipeline`, valores en el
  código).
- En `demos/demo_mission.py` el controlador y la FSM corren cada
  `CTRL_EVERY × DT = 5 × 0.05 = 0.25 s` (el comentario del demo dice 0.5 s;
  es incorrecto).

## 2. Controladores de actitud

Estado (2026-10-25):

- **Roll:** tu ley, parte de PID no lineal con compensación de gravedad
  (entrada 14 de la bitácora; faltan los términos d_φ, m_φ y Σ_ψ).
  Ganancias sintonizadas por el autor con control a 100 Hz.
- **Pitch:** el provisional de la sesión anterior; según el autor es casi
  igual a su PID no lineal y sirve así (entrada 12).
- **Yaw:** tu ley, PID con el roll como control virtual (entrada 15).

Ganancias: constantes al inicio de `autopilot.py` (líneas 47-55). Cada
lazo es una función (sección 5).

**Pitch: PID** (`KP_P = 0.015`, `KD_P = 0.08`, `KI_P = 0.005` (el autor
la subió de 0.003), `I_MAX_P = 0.005`)

```
e      = θ − θ_ref
ė      = θ̇ − θ̇_ref                     (θ̇ de la cinemática, NO q)
I      = sat(I + KI·e·dt, ±I_MAX)       (aporte integral, en m de x_g)
x_g    = (1/cos θ) · (KP·e + KD·ė + I − z_g0·sin θ)
rc_cmd = m_tot · x_g / m_lin
```

- La salida es la posición deseada del centro de gravedad, convertida a
  posición de la masa móvil.
- `−z_g0·sin θ`: compensación de gravedad (z_g0 se calcula de los parámetros).
- **θ̇ no es q.** q es la velocidad angular del cuerpo alrededor de y_b;
  θ̇ sale de η̇ = J(η)ν: θ̇ = q·cos φ − r·sin φ (`euler_rates(x)`). Solo
  coinciden con φ = 0.
- θ̇_ref: diferencia hacia atrás de la referencia moldeada en un periodo
  de control (la referencia es suave por el moldeado).
- Integral: se guarda directamente el **aporte** I = KI·∫e (en `mem[0]`,
  metros de x_g). `KI_P` fija qué tan rápido se mueve y `I_MAX_P` cuánto
  puede aportar como máximo; ese tope es todo el anti-windup (decisión del
  autor, entrada 12 de la bitácora). Como el estado se satura, no se
  acumula integral "escondida" detrás del tope.
- Con roll el momento de gravedad es −W·(z_g·sin θ + x_g·cos θ·cos φ); la
  ley ignora el cos φ (≈ 3 % con 15° de roll).

**Yaw: PID del autor → roll de referencia** (sintonizado por el autor,
2026-10-25: `KP_Y = 0.8`, `KD_Y = 0.2`, `KI_Y = 0.01`, `I_MAX_Y = 0.02`
rad, `ROLL_MAX = 0.8` rad)

```
e        = wrap(ψ − ψ_ref)                (medido − referencia, camino corto)
ė        = ψ̇ − ψ̇_ref                       (ψ̇ de la cinemática, NO r)
I        = sat(I + KI·e·dt, ±I_MAX)        (aporte integral, en rad de roll)
roll_ref = sat((−KP·e − KD·ė − I) · turn_sign, ±ROLL_MAX)
```

- `wrap(a) = atan2(sin a, cos a)`: el error siempre es el del camino
  corto. El moldeado de referencias también gira por el camino corto: el
  rumbo crudo se lleva a ±180° del estado del limitador antes de limitarlo
  (`sim.py::make_ref_pipeline`), así que 170° → −170° son 20°, no 340°.
  El rumbo moldeado es continuo y puede salir de ±180°.
- ψ̇ = (q·sin φ + r·cos φ)/cos θ (`euler_rates(x)[2]`).
- `turn_sign` (signo de giro, `rollyawdir` en Simulink; antes se llamaba
  `cg`): en misión viene de la FSM, +1 en DIVE, −1 en CLIMB, 0 en los
  demás estados. En `02_simulacion.py` (modo `pitch_roll_yaw`) se calcula
  igual a partir del pitch de la tabla: bajada +1, subida −1, pitch 0 → 0.
  Invierte el sentido del giro entre bajada y subida (ver
  [MODELO.md](MODELO.md), sección 6). Con `turn_sign = 0` no hay control
  de rumbo (el integrador sigue integrando, hasta su tope).
- `I_MAX_Y = 0.02` rad = 1.15° de roll (el autor lo bajó desde 0.1 rad).

**Roll: ley del autor, parte de PID no lineal con compensación de
gravedad** (ganancias sintonizadas por el autor con control a 100 Hz:
`KP_R = 0.008`, `KD_R = 0.02`, `KI_R = 0.01`, `I_MAX_R = 0.001`)

```
e      = φ − φ_ref                     (medido − referencia, igual que pitch)
ė      = φ̇ − φ̇_ref                     (φ̇ de la cinemática, NO p)
I      = sat(I + KI·e·dt, ±I_MAX)       (aporte integral, en m de u)
u      = (1/cos φ) · (KP·e + KD·ė + I − z_g·sin φ)
γ_cmd  = asin( sat(u·m_tot / (m_rot·R_rot), ±1) )
```

- **u es el "y_g" de tu ley: CG hacia babor**, o sea −y_g del marco del
  cuerpo (la dirección en que va la masa con γ > 0). Lo decide el término
  de gravedad: para sostener φ > 0 el CG va a estribor, y la ley da
  −z_g·tan φ. Con u, el roll tiene la misma forma que pitch,
  M_x = −W·cos θ·(z_g·sin φ + u·cos φ). Por eso sin γ va con signo + (tu
  `fcn` de Simulink lleva − porque ahí y_g es el de estribor). Detalle y
  pruebas: [DISENO_ROLL.md](DISENO_ROLL.md), sección 8.
- u → γ: la masa rotatoria está en y_rot = −R_rot·sin γ, así que
  u = m_rot·R_rot·sin γ / m_tot. m_tot, m_rot, R_rot y z_g salen de los
  parámetros (`nominal_mass`: sin `m_dist`, γ = 0). Autoridad máxima:
  |u| ≤ m_rot·R_rot/m_tot = 6.0 mm.
- φ̇ = p + (q·sin φ + r·cos φ)·tan θ (`euler_rates(x)[0]`).
- Faltan los términos −(d_φ/W)·φ̇, −(m_φ/W)·φ̈_d y Σ_ψ.
- Con las ganancias de Simulink el sobrepaso aumenta (la ley cancela la
  rigidez del péndulo que el PID simplificado tenía gratis): hay que
  sintonizar. Tabla de pruebas en [DISENO_ROLL.md](DISENO_ROLL.md).
- Si prendes la parte D en modo `pitch_roll`, el escalón de roll de la
  tabla da un pico de φ̇_ref (15°/0.25 s = 1 rad/s) en una muestra.

**Flotabilidad: sin lazo.** `dB = BUOY_SIGN · Buoy_ref`, con
`BUOY_SIGN = −1` (**confirmado** contra el diagrama de Simulink: ganancia
−1 entre `Buoy_ref` y el VBD; BITACORA, entrada 17). Convierte la convención de V2
(+3 bajar, −3 subir) a la física del VBD (dB negativo = menos flotación =
baja). El VBD satura en ±4.93 N (carrera completa).

## 3. FSM de waypoints

### 3.1 Organización: una FSM por archivo

```
gliderjax/navigation.py        lo común a todas las FSM: sensores, salidas,
                               GPS, misiones y su conversión a NED
gliderjax/fsm/__init__.py      registro: fsm.make("simulink", mision)
gliderjax/fsm/simulink.py      tu FSM (Simulink, versión 2026-10)
gliderjax/fsm/dive_models.py   modelos de inmersión (predicción y plan)
```

**Contrato de una FSM** (`navigation.py`): `fsm.make(nombre, mision)`
devuelve un `Navigator` con:
- `carry0`: la memoria inicial (las variables `persistent` de Simulink),
  con un nombre por entrada (`carry_names`);
- `step(carry, sens)` → `(carry, out)`, que corre en cada tick del
  controlador.

`sens` (`Sensors`) es lo que la FSM puede leer:
- `fix` [lon°, lat°], último fix GPS;
- `gps_ok`: 1 si el fix es fresco;
- `depth`, `roll`, `pitch`, `yaw`, `t`.

`out` tiene cinco salidas:
- `pitch_ref`, `yaw_ref`;
- `buoy_ref`, en la convención de la FSM (+3 baja); el autopiloto lo
  multiplica por `BUOY_SIGN = −1`, como la ganancia −1 de Simulink;
- `turn_sign`: +1 bajada, −1 subida, 0 sin control de rumbo;
- `mission_complete`.

La FSM no ve la verdad; solo ve los sensores.

**Cómo hacer una FSM alternativa:**
- **Misma lógica, otro modelo de inmersión** (por ejemplo, contar el giro
  al calcular la profundidad): escribir un `DiveModel` en
  `fsm/dive_models.py` y pasarlo:
  `fsm.make("simulink", mision, dive_model=MI_MODELO)`. Un `DiveModel`
  tiene dos funciones:
  - `displacement(yaw_inicio, yaw_ref, prof, cfg)`: el desplazamiento
    horizontal esperado de una inmersión completa. Lo usa el estimador de
    deriva.
  - `plan(vec_al_objetivo, D_norm, yaw_actual, cfg)`: devuelve
    `(yaw_ref, prof)`.

  La FSM guarda `yaw_at_plan` (el rumbo al planear) para los modelos que lo
  necesiten. El de Simulink, `STRAIGHT_GLIDE`, no lo usa.
- **Otra lógica:** copiar `fsm/simulink.py` a un archivo nuevo, cambiarlo y
  registrarlo en `FSMS` (`fsm/__init__.py`). La memoria puede tener otro
  tamaño: el autopiloto la guarda al final de `ctrl_state`.
- **Constantes:** cualquier campo de `Config` se cambia al crear la FSM,
  por ejemplo `fsm.make("simulink", mision, pitch_down_rad=-0.5)`.

En la telemetría, `autopilot.waypoint_signals(tel, nav)` da todo por
nombre: el fix, las cinco salidas y cada entrada de la memoria
(`glider_state`, `current_wp_index`, `D_norm_n`, ...).

### 3.2 Tu FSM (`fsm/simulink.py`)

Port de `fcn_Glider_State_Machine`, versión actual
(`referencia_simulink/fsm_y_geodesia.m`), con los mismos nombres de
constantes y variables. Diferencias deliberadas:
1. `lla2ecef` con `sin(lon)` en `y`. Simulink tiene `cos(lon)`, un error
   que deforma el mapa de la FSM (BITACORA, entrada 17).
2. DONE con flotabilidad positiva: `buoy_done_ref = −3`. En Simulink queda
   en 0 (neutra). Decisión del autor, 2026-10-24.
3. El modelo de inmersión es intercambiable. El de Simulink es el default.
4. Pitch por tramo opcional (`leg_pitch`): las inmersiones hacia el
   waypoint i usan su propio pitch de bajada y de subida. Sin él, es la FSM
   de Simulink. Se da en la tabla de `03_mision.py` ([MISION.md](MISION.md)).

Constantes (`Config`, en `fsm/simulink.py`):

| Constante | Valor | Significado |
|---|---|---|
| `acceptance_radius_m` | 90 m | radio de llegada al waypoint |
| `surface_wait_time_s` | 60 s | espera en superficie antes de planear (FIX_PLAN) |
| `bottom_wait_time_s` | 10 s | espera en el fondo |
| `surface_stabilize_time_s` | 10 s | estabilización al salir (SURFACE_WAIT) |
| `pitch_antenna_rad` | −1.1 rad (−63°) | postura de antena |
| `pitch_down_rad` / `pitch_up_rad` | ∓0.6109 rad (∓35°) | pitch de bajada / subida |
| `buoy_down_ref` / `buoy_up_ref` | +3 / −3 | flotabilidad (convención de la FSM) |
| `buoy_done_ref` | −3 | flotabilidad en DONE (Simulink: 0) |
| `max_dive_depth_m` | 200 m | profundidad máxima planeada |
| `initial_turn_depth_m` | 2 m | de SUBMERGE a DIVE_TURN |
| `depth_epsilon_m` | 2 m | margen para llegar a la profundidad planeada |
| `surface_epsilon_m` | 1 m | profundidad que cuenta como superficie |
| `flare_depth_m` | 3.5 m | inicio del flare |
| `pitch_flare_target` | −1.1 rad | pitch al final del flare |
| `flare_duration_s` | 10 s | duración de la rampa del flare |
| `min_estimation_depth_m` | 15 m | profundidad mínima para estimar deriva |
| `baseline_glide_ratio` | 1.4 | razón de planeo supuesta (k) |
| `alpha_filter` | 0.8 | filtro del estimador de deriva |

| # | Estado | pitch_ref | yaw_ref | Buoy | signo de giro | Sale cuando |
|---|---|---|---|---|---|---|
| 1 | FIX_PLAN | −63° | planeado | −3 (positiva) | 0 | pasan 60 s → SUBMERGE, o DONE si no quedan waypoints |
| 2 | SUBMERGE | −35° | actual | +3 (negativa) | 0 | prof. > 2 m → DIVE_TURN |
| 3 | DIVE_TURN | −35° | planeado (camino corto) | +3 (negativa) | **+1** | prof. planeada − prof. < 2 m → BOTTOM_WAIT |
| 4 | BOTTOM_WAIT | 0° | planeado | 0 (neutra) | 0 | pasan 10 s → CLIMB |
| 5 | CLIMB | +35°; flare desde 3.5 m: rampa a −63° en 10 s | planeado | −3 (positiva) | **−1** | prof. ≤ 1 m → SURFACE_WAIT |
| 6 | DONE | −63° | actual | −3 (positiva) | 0 | nunca (final) |
| 7 | SURFACE_WAIT | −63° | planeado | −3 (positiva) | 0 | pasan 10 s → FIX_PLAN |

En FIX_PLAN, al cumplirse los 60 s:
1. Lee el fix.
2. Estima la deriva, si la inmersión pasó de 15 m: compara dónde salió con
   lo que predice el modelo de inmersión.
3. Revisa si llegó al waypoint.
4. Planea el rumbo y la profundidad. Con `STRAIGHT_GLIDE`:
   `prof = clip(dist_corregida / (2k), 4 m, 200 m)`.

Si en ese tick se acaban los waypoints, la salida es la de default
`[0, yaw, 0, 0, 0]`: es el `return` de Simulink.

Misiones predefinidas en `navigation.py`: `MISSION_TRIANGLE` (la de tu
Simulink), `MISSION_SQUARE7`, `MISSION_SQUARE_3KM`, `MISSION_LAWNMOWER`.
Formato `[lon, lat, h]`, el orden de Fossen y de tu Simulink.
`check_mission` rechaza una latitud fuera de ±90°: en el Golfo, lat/lon
invertidas dan "lat = −95" y se detecta.

**Ejemplo, triángulo (2026-10-25, perfil `simulink` + geodesia, control a
100 Hz, GPS ideal):**
- SUBMERGE en t = 61 s, prof. planeada 139 m;
- WP2 a 16 m, en t = 969 s;
- WP3 a 48 m, en t = 1897 s;
- DONE en t = 1957 s; queda flotando a z = 0.85 m, pitch −63°.

## 4. Problemas conocidos

1. **GPS ideal.** Como en Simulink, siempre hay fix con la posición
   verdadera (`gps_gate_depth = inf`), y la FSM solo lo lee en superficie
   (FIX_PLAN). Sin ruido y sin revisar si la antena sale del agua. Decisión
   del autor, 2026-10-25 (BITACORA, entrada 19). Con la compuerta anterior
   (0.7 m) nunca había fix: en postura de antena (−63°), z ≈ 0.85-1.8 m.
2. **Deriva falsa por el giro.** Sin corriente, el estimador de deriva da
   `D_norm` ≈ 0.1-0.2 m/m. El modelo recto no cuenta el giro al inicio de
   la inmersión. Es justo lo que mejoraría un modelo de inmersión con giro
   (sección 3.1).
3. **Ruido GPS no implementado** (`gps_sigma_m` existe pero es TODO).
4. **Razón de planeo fija.** El pitch ya se da por tramo (decisión del
   autor, 2026-10-24: sin pitch automático en el actuador). Pero el modelo
   de inmersión recto supone k = 1.4 con cualquier pitch: a −25° el
   planeador avanza más y se pasa del waypoint ([MISION.md](MISION.md),
   pruebas). Pendiente: un modelo que tome k del pitch del tramo.
5. **El integrador de yaw sigue integrando con `turn_sign = 0`**
   (superficie, fondo), hasta su tope.

## 5. Cómo implementar tu controlador

Todo está en `gliderjax/autopilot.py`. Hay **una función por lazo**; para
poner tu control, reemplazas el cuerpo de la que corresponda:

| Función | Línea | Recibe | Devuelve |
|---|---|---|---|
| `pitch_control(x, params, pitch_ref, pitch_ref_dot, mem, dt)` | 97 | referencia de pitch [rad] y su derivada [rad/s] | `rc_cmd` [m] |
| `roll_control(x, params, roll_ref, roll_ref_dot, mem, dt)` | 112 | referencia de roll [rad] y su derivada | `gamma_cmd` [rad] |
| `yaw_control(x, params, yaw_ref, yaw_ref_dot, turn_sign, mem, dt)` | 135 | referencia de rumbo [rad], su derivada y signo de giro | `roll_ref` [rad] |

Las tres devuelven además `mem` actualizada. `attitude_step` (línea 151)
las conecta y calcula las derivadas de las referencias; no hace falta
tocarla.

**Entradas comunes:**

| | Contenido |
|---|---|
| `x` | estado verdadero (26): `x[2]` z, `x[3]` φ, `x[4]` θ, `x[5]` ψ, `x[6:9]` u, v, w, `x[9]` p, `x[10]` q, `x[11]` r [m, rad, m/s, rad/s]. **p, q, r son del cuerpo, no las derivadas de los ángulos**: para φ̇, θ̇, ψ̇ usa `euler_rates(x)` (línea 82) |
| `*_ref_dot` | derivada de la referencia [rad/s]: diferencia hacia atrás en un periodo de control. Pitch y rumbo vienen moldeados (suaves); en modo `pitch_roll` el escalón de roll de la tabla da un pico de escalón/dt en una sola muestra |
| `params` | todo lo del archivo de configuración (`params.m_lin`, `params.r_lin0`, ...) |
| `mem` | la **memoria** de ese lazo: `N_MEM = 4` números que se guardan entre llamadas (integradores, valor anterior para una derivada, estado de un filtro). Empieza en 0 |
| `dt` | periodo del controlador [s] (`PERIODO_CONTROL`) |

Las referencias llegan **ya moldeadas**: pitch ±0.1 rad/s → 1/(5s+1);
rumbo ±0.03 rad/s → 1/(5s+1) (por el camino corto); roll sin moldeado.

**Cómo corren:** cada `PERIODO_CONTROL`; el comando se mantiene entre
ejecuciones (ZOH) y luego pasa por las cadenas de los actuadores y sus
saturaciones (rc ±0.10 m, γ ±90°). El anti-windup es responsabilidad del
controlador.

**Modos de prueba** (`02_simulacion.py`):

| `MODO` | Qué lazos corren | Tabla |
|---|---|---|
| `"pitch_roll"` | pitch y roll, cada uno con su referencia; el de yaw no corre | `FASES_PITCH_ROLL`: t, pitch_ref [°], roll_ref [°], dB [N] |
| `"pitch_roll_yaw"` | pitch; yaw → roll_ref → roll; signo de giro automático (bajada +1, subida −1) | `FASES_PITCH_ROLL_YAW`: t, pitch_ref [°], rumbo_ref [°], dB [N] |

Plan: sintonizar pitch y roll con `"pitch_roll"`, luego el de yaw con
`"pitch_roll_yaw"`, y después la misión por waypoints (FSM).

**Reglas de JAX** (la función se compila):

1. Sin `if` de Python con valores del estado: `jnp.where(cond, a, b)`;
   para saturar, `jnp.clip(u, -lim, lim)`.
2. Usa `jnp` (`jnp.sin`, `jnp.abs`, ...), no `np` ni `math`.
3. Lo que deba recordarse va en `mem` (`mem.at[0].set(valor)` devuelve una
   copia modificada); nada en variables globales.
4. `print` no funciona adentro: usa `jax.debug.print("e = {}", e)`, o
   grafica la memoria desde la telemetría (abajo).

**Signos:** rc > 0 baja la nariz; γ > 0 da roll negativo (ala izquierda
abajo); θ > 0 nariz arriba; φ > 0 ala derecha abajo; ψ rumbo NED (0 =
norte, + hacia el este).

**Memoria y referencias en la telemetría** (`tel["ctrl_state"]`, o
`scope.scope(tel, ["ctrl_state:12"])`):

| Índices | Contenido |
|---|---|
| 0–3 | memoria de pitch (0 = aporte integral KI·∫e del PID provisional, en m de x_g) |
| 4–7 | memoria de roll (0 = aporte integral KI·∫e, en m de u = −y_g) |
| 8–11 | memoria de yaw (0 = aporte integral KI·∫e, en rad de roll) |
| 12, 13, 14 | referencias que usaron los lazos: pitch, roll, yaw [rad] (también el valor anterior para las derivadas) |
| 15–23 | estados del moldeado de referencias |
| 24–25 | solo misión: fix GPS [lon°, lat°] |
| 26–30 | solo misión: salidas de la FSM (`NAV_OUT_NAMES`) |
| 31– | solo misión: memoria de la FSM (`carry_names`; el tamaño depende de la FSM) |

**Para no perder el provisional:** está en git (`git diff
gliderjax/autopilot.py` muestra tus cambios; `git show
HEAD:gliderjax/autopilot.py` la versión guardada).
