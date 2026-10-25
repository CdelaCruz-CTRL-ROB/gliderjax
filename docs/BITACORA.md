# Bitácora

Registro de pasos, experimentos y decisiones. Lo más reciente arriba.
El detalle de cada cambio al código está en git (`git log`, `git show <id>`).

## Pendientes (lista viva)

- [x] Saturación tanh como switch, apagada por defecto (= Simulink).
- [x] Script de validación en lazo abierto, sin geolocalización.
- [x] `scope.py`: guardar/cargar telemetría, comparar corridas, vistas nuevas.
- [x] Control de versiones con git.
- [ ] Validar contra una corrida de Simulink exportada a CSV (ver
      [VALIDACION.md](VALIDACION.md)).
- [x] Comparar contra `Mbr+Ma fcn` y `calculate_coriolis_total_fossen`.
- [x] Comparar contra `fcn_VBD_actuator`, `fcn_buoyancy_surface` y
      `fcn_r_lin`.
- [x] Inercia de la masa rotatoria: R·I·Rᵀ (decisión del autor).
- [x] Masa añadida: solo la diagonal (decisión del autor).
- [x] Archivo de configuración `config/cidesi.toml`.
- [x] Superficie: `B_scale` multiplica a B, `Δx_b` se suma a x_b
      (confirmado).
- [x] Flotabilidad separada de las masas: `rho·g·V_displaced`
      (decisión final del autor; neutralidad a mano).
- [x] Script de balance estático y cero virtual (`00_balance_estatico.py`).
- [x] Script de simulación manual con solver, condiciones iniciales y
      comparación con la corrida anterior (`02_simulacion.py`).
- [x] Vector de restauración verificado (`fcn_restauracion.m`).
- [x] Cadenas de actuadores verificadas (autor).
- [x] Switch directo / control en `02_simulacion.py`.
- [x] Perturbaciones: solo la masa de perturbación `m_dist` (se quitan
      `distxlin`, `dist_xg`, `dist_yg`); error de cero equivalente informativo.
- [x] `cg` → `turn_sign` / "signo de giro".
- [x] Error de calibración en γ: no se agrega; una `m_dist` lateral da el
      mismo efecto (ver [MODELO.md](MODELO.md), sección 2.5).
- [ ] Barridos de parámetros: correr varias variantes y superponerlas
      ("mover todo y ver qué da"). Puede salir de `02_simulacion.py`.
- [ ] ¿Escalar los coeficientes hidrodinámicos con ρ/ρ_CFD al cambiar
      `rho`? Hoy no se escalan.
- [x] Confirmar `BUOY_SIGN` (convención de la FSM V2 → VBD): −1
      (entrada 17).
- [x] Actualizar la FSM del port a la versión actual de Simulink:
      constantes, flare por tiempo, pitch de antena en SURF (entrada 18).
- [x] **GPS: ¿compuerta o no?** Sin compuerta, como Simulink
      (`gps_gate_depth = inf`); sin ruido ni antena por ahora (entrada 19).
- [ ] Futuro: GPS con ruido (`gps_sigma_m`) y fix solo con la antena fuera
      del agua (necesita la posición de la antena).
- [ ] Sintonizar la corrección del estimador de deriva: con corriente
      corrige peor que en Simulink (autor, 2026-10-25). Ver también los
      modelos de inmersión (abajo).
- [ ] Modelo de inmersión con giro (`fsm/dive_models.py`): sin corriente,
      el modelo recto deja una deriva falsa de 0.1-0.2 m/m (entrada 18).
- [ ] Modelo de inmersión con k según el pitch del tramo: con ∓25° el
      planeador se pasa 147 m del WP2 (entrada 20). Para eso `plan` y
      `displacement` tendrían que recibir el pitch del tramo (hoy no lo
      reciben).
- [ ] ¿De dónde sale la condición inicial del integrador `ECEF_XYZ` en
      Simulink? (entrada 17)
- [ ] Pasar al archivo de configuración las ganancias de control y el
      moldeado de referencias (cuando lleguen tus controladores).
- [x] Script de waypoints: pitch de cada inmersión como dato de la misión
      (sin pitch automático en el VBD). Por tramo (entrada 20).
- [ ] Revisar si sign(α) en KMR es necesario (la inversión del giro ocurre
      sin él).
- [x] DONE con flotabilidad positiva. Simulink también pide 0 (entrada
      17); queda la decisión del autor: positiva (`buoy_done_ref`, entrada 18).
- [x] Revisar `FLARE_DEPTH` / `SURF_EPS` contra V2: en Simulink el flare
      es por tiempo desde 3.5 m (entrada 17); se resuelve al actualizar la
      FSM.
- [ ] Diseño de roll: resuelto en la implementación (la salida de la ley
      se aplica como u = −y_g, entrada 14). Falta revisar en el diseño el
      signo del término de y_g del modelo de roll (Fossen g₄ lleva −y_g;
      ver [DISENO_ROLL.md](DISENO_ROLL.md), sección 8).
- [ ] Roll: términos −(d_φ/W)·φ̇, −(m_φ/W)·φ̈_d y Σ_ψ de la ley; sintonizar
      con la compensación de gravedad (ver [DISENO_ROLL.md](DISENO_ROLL.md),
      sección 8).
- [x] Reemplazar controladores provisionales por los del autor: roll
      (entradas 13 y 14), yaw (entrada 15); pitch se queda el provisional
      (entrada 12).
- [x] Yaw: sobrepaso de ~13° por carga del integrador; sintonizar
      `I_MAX_Y` / `KI_Y` o la razón del moldeado de rumbo (entrada 15).
      Sintonizado por el autor (entrada 16).
- [ ] Waypoints: integrador de yaw con `turn_sign = 0` (superficie,
      fondo) sigue integrando hasta su tope (entrada 15).
- [x] Script de misión por waypoints: `escenarios/03_mision.py` (entrada 20).
- [ ] Cheat sheet: "quiero cambiar X → dónde" (primera versión en
      [README.md](README.md)).
- [ ] Futuro: formación de N planeadores.

---

## 2026-10-25 (21) — Trayectorias para comentar/descomentar; mapa con tu formato

Pedido del autor:
- trayectorias que se eligen comentando y descomentando, como su lista de
  Simulink;
- en el mapa, el punto de inflexión de cada inmersión, con la profundidad
  que alcanzó;
- flechas de corriente con la simbología de su figura
  (`referencia_simulink/figura_waypoints.png`), en una sola corrida, sin comparar con y
  sin estimador.

Cambios en `escenarios/03_mision.py`:
- `WAYPOINTS` trae las trayectorias de su lista como bloques:
  - triángulo (activo);
  - cuadro de 7 puntos;
  - un punto al norte;
  - cuadro de 3 km;
  - barrido.

  El origen siempre es la primera fila.
- Numeración como en su lista: el origen es **WP0**. En consola y figuras
  los objetivos son WP1, WP2, … La FSM sigue numerando desde 1 por dentro.
  "Alcanzados" ya no cuenta el origen.
- `10_mapa.png` con su formato:
  - Norte X horizontal, Este Y vertical, letra con serifas;
  - línea roja = trayectoria;
  - flechas azul claro = corriente uniforme del TOML;
  - X roja con círculo punteado = target;
  - ▼ = inflexión, con su profundidad;
  - ▲ cian = superficie;
  - ★ verde = inicio.

  La leyenda va fuera del área de la gráfica, y los waypoints repetidos
  llevan una sola etiqueta ("WP2, WP6").
- `check_mission`: el error da el WP (origen = WP0) y la fila.

Prueba con corriente (solo en la prueba; el TOML sigue sin corriente):
cuadro de 7 puntos con N −0.05, E +0.08 m/s. Completa 6/6, DONE en
t = 7685 s, 8 inmersiones, de las cuales dos son cortas, de 29 y 46 m, para
entrar al radio del WP3 y del WP6. Sin corriente, el triángulo da lo mismo
que antes (DONE en t = 1957 s).

---

## 2026-10-25 (20) — Script de misión: `escenarios/03_mision.py`

Guía en [MISION.md](MISION.md). Mismo estilo que `02_simulacion.py`:
- waypoints (lon, lat), con pitch opcional por tramo;
- FSM y modelo de inmersión por nombre;
- `FSM_AJUSTES` para cambiar constantes;
- figuras de mapa, perfil, actitud y deriva;
- una fila por inmersión en consola;
- comparación de entradas con la corrida anterior.

**Duración** (pedido del autor): `"tiempo"` (corta en `T_FINAL`) o
`"mision"` (hasta DONE + 20 s, como el `stop` de Simulink, con aviso de
TIMEOUT en `T_MAX`). Para eso:
- `sim.make_resumable_simulator`: simula por bloques que siguen donde quedó
  el anterior;
- `sim.make_simulator` comparte con él el código del paso (`_make_frame`).

Verificado:
- `02_simulacion.py` da lo mismo que antes del cambio (diferencia 0.0 en
  las 30 señales);
- 3 bloques de 400 s dan lo mismo que un bloque de 1200 s (diferencia 0.0).

Otros cambios en la librería:
- `fsm/simulink.py`: `leg_pitch`, pitch de bajada y subida por waypoint
  (por tramo); el flare arranca del pitch de subida del tramo. Sin él,
  igual que antes.
- `fsm/dive_models.py`: `DIVE_MODELS`, los modelos por nombre.
- `Navigator.config`: las constantes de la FSM, para imprimir y graficar.
- `make_waypoint_controller(..., init_refs)`: el moldeado arranca en la
  actitud inicial, como en `make_attitude_controller`.
- `check_mission`: el error dice el WP con numeración desde 1.

Pruebas, todas en el triángulo de Simulink:

| Caso | Resultado |
|---|---|
| `"mision"` | DONE en t = 1957 s, simula hasta 1977 s, 14 s de cómputo |
| `"mision"`, `T_MAX = 1200` | TIMEOUT, 2 de 3 |
| `"tiempo"`, `T_FINAL = 1200` | tiempo cumplido sin terminar, 2 de 3 |
| WP2 con ∓25° | completa en 3456 s con 3 inmersiones: la 1a se pasa 147 m del WP2 porque el modelo recto supone k = 1.4, y a −25° el planeador avanza más |
| lat/lon invertidas | error con el WP |

En superficie la flotabilidad neta oscila (12_actitud.png). Es el modelo
de superficie de 2 puntos (`fcn_buoyancy_surface`).

---

## 2026-10-25 (19) — GPS ideal, como Simulink

Decisión del autor: leer el GPS en superficie, sin considerar por ahora
ruido ni si la antena sale o no. `config/cidesi.toml`:
`gps_gate_depth = inf` (antes 0.7). Siempre hay fix con la posición
verdadera, como en Simulink, y la FSM solo lo lee en FIX_PLAN.

Verificación: el triángulo, ahora directo del archivo de configuración
(sin cambios en el script), da lo mismo que la prueba sin compuerta de la
entrada 18: DONE en t = 1957 s.

`demos/demo_mission.py` (cuadro de 3 km, modelo completo, control a
0.25 s): completa, 5/5 waypoints, DONE en t = 28327 s. Con la compuerta
llegaba a 1/5. D_norm final [−0.049, +0.046] m/m.

---

## 2026-10-25 (18) — FSM: una por archivo; tu FSM actual portada

Pedido del autor: implementar la FSM actual, pero dejar fácil crear
alternativas, por ejemplo una que cuente el giro en el modelo simplificado
que calcula la profundidad de la inmersión.

### Organización (guía en CONTROL_Y_MISION.md, sección 3.1)

- `navigation.py`: queda solo lo común:
  - `Sensors`, lo que una FSM puede leer: fix, gps_ok, depth, roll, pitch,
    yaw, t;
  - las 5 salidas (`nav_out`);
  - `Navigator`, el contrato;
  - GPS;
  - misiones, con `check_mission`, que detecta lat/lon invertidas.
- `fsm/__init__.py`: registro, `fsm.make("simulink", mision, **opciones)`.
- `fsm/simulink.py`: tu FSM, con los nombres de Simulink. Constantes en
  `Config` y memoria en `Carry`, ambos con nombre.
- `fsm/dive_models.py`: `DiveModel(displacement, plan)`. El de Simulink es
  `STRAIGHT_GLIDE`.
- `autopilot.make_waypoint_controller(nav, dt)` acepta cualquier FSM.
  `ctrl_state`:
  - [24:26] fix;
  - [26:31] salidas de la FSM;
  - [31:] memoria (tamaño libre).

  `waypoint_signals(tel, nav)` lee todo por nombre.
- `demos/demo_mission.py` adaptado a la nueva interfaz.

### Cambios respecto al port anterior ("V2")

Ahora sigue a `referencia_simulink/fsm_y_geodesia.m`:
- espera en superficie de 60 s;
- `depth_epsilon` 2 m;
- `min_estimation_depth` 15 m;
- flare por tiempo desde 3.5 m (rampa de +35° a −63° en 10 s);
- pitch de antena en SURFACE_WAIT;
- en el tick en que planea, FIX_PLAN entrega el rumbo anterior (como
  Simulink);
- DONE con flotabilidad positiva (`buoy_done_ref = −3`, decisión del
  autor).

`lla2ecef` sigue corregido. `mission_complete` ya no es memoria: es una
salida.

### Verificación (triángulo de tu Simulink, perfil `simulink` + geodesia, dt = 0.005 s, control a 100 Hz)

- **Con `gps_gate_depth = 0.7` (el valor del TOML): no llega.** En
  superficie, con pitch −63°, el origen del cuerpo queda a z ≈ 0.85-1.8 m.
  Nunca hay fix fresco, la FSM planea con el fix del arranque y el
  planeador se va al noreste (D_norm ≈ −1.9 m/m). Lo mismo con el perfil
  `completo`, y en el demo (1/5 waypoints). Antes funcionaba porque el port
  viejo ponía pitch 0° en SURF y el planeador quedaba más horizontal.
- **Sin compuerta (como Simulink, solo en la prueba): llega.** SUBMERGE
  en t = 61 s (prof. planeada 139 m), WP2 a 16 m en t = 969 s, WP3 a 48 m
  en t = 1897 s, DONE en t = 1957 s. Queda flotando con z = 0.85 m y pitch
  −63°.
- Sin corriente, el estimador da D_norm = [+0.004, −0.091] tras la 1a
  inmersión y [−0.006, +0.197] tras la 2a. Es deriva falsa: el modelo
  recto no cuenta el giro (2a inmersión: empieza con rumbo 45° y gira a
  −42°). Ejemplo del modelo de inmersión con giro que propone el autor.
- `02_simulacion.py` (modos sin FSM) sin cambios: corre igual.

---

## 2026-10-25 (17) — Geodesia y FSM de Simulink: comparación con el port

El autor dejó en `sim/simulink/` su código actual de geodesia y FSM y
capturas del diagrama. Copiados tal cual a `docs/referencia_simulink/`:
`fsm_y_geodesia.m`, `diagrama_geodesia.png`, `diagrama_fsm.png`,
`diagrama_fsm_2.png`.

### Geodesia en Simulink

`eta_dot(1:3)` (NED) → `fcn_get_R_n_e` (ECEF_DOT = R_n_e(lon, lat)·ṗ) →
integrador `ECEF_XYZ` → `fcn_ecef2lla` → `lla_rad_vec` (vuelve a
`fcn_get_R_n_e`) y `lla_deg_lon_lat` (→ FSM). El port hace lo mismo
(`dynamics.py`, líneas 354-357).

**Orden lon/lat: consistente, no es error.** En todo el código el vector
es `[lon, lat, h]`, la notación de Fossen (l, μ, h) con R_n^e(l, μ). La
matriz `R_n_e` es la de Fossen con l = lon y μ = lat. Las entradas
coinciden en todos los usos: waypoints `[lon, lat, 0]`, origen, fix,
`fcn_get_R_n_e`. Solo hay dos nombres engañosos:
- el comentario de `fcn_ecef2lla` dice `[lat_deg, lon_deg]`, pero la
  salida es `[lon; lat]`;
- la entrada `ECEF` de `fcn_get_R_n_e` recibe en realidad `lla_rad_vec`.

**Error real: `fcn_lla2ecef` (línea 415) calcula `y` con `cos_lon`; debe
ser `sin_lon`.** Esa función solo la usa la FSM, para el origen, los
waypoints y el fix. La posición integrada usa la cinemática y
`fcn_ecef2lla`, que están bien. Efecto: la FSM ve un mapa NED
distorsionado. Cerca del origen, NED_FSM = A·NED_real con
A = [[0.902, 0.361], [0.030, 0.891]]:

| Rumbo real | La FSM lo ve como | Escala de distancia |
|---|---|---|
| 0° | 1.9° | 0.90 |
| 45° | 36.1° | 1.10 |
| 90° | 67.9° | 0.96 |
| 135° | 122.1° | 0.72 |

Misión triángulo: el WP2 real está a 391.8 m con rumbo 45.1°; la FSM lo
pone a 433.1 m con rumbo 36.1°. El planeador apunta con errores de rumbo
de hasta ~22°. El radio de llegada de 90 m se mide en metros
distorsionados (de 0.72 a 1.10). Probablemente el estimador de deriva
absorbe parte del error como si fuera corriente, y al replanear en cada
fix se corrige: por eso la misión llega.

El port ya tenía corregido `lla2ecef` (`geodesy.py:12`, "Corrected: y uses
sin_lon"). Se mantiene corregido (regla del autor: no replicar errores de
Simulink).

Con el error, el ECEF del origen da lon = −135°. Por eso la condición
inicial del integrador `ECEF_XYZ` en Simulink tiene que venir de otro lado.
Pendiente preguntar de dónde.

### FSM: diferencias entre la versión actual de Simulink y el port ("V2")

| | Simulink actual | Port |
|---|---|---|
| `surface_wait_time_s` | 60 s | 30 s |
| `depth_epsilon_m` | 2 m | 4 m |
| `flare_depth_m` | 3.5 m | 1 m |
| `min_estimation_depth_m` | 15 m | 20 m |
| Flare | por tiempo: al pasar de 3.5 m, rampa de +35° a −63° (`pitch_flare_target`) en 10 s | por profundidad hacia 0°, sin efecto (`FLARE_DEPTH = SURF_EPS`) |
| pitch en SURFACE_WAIT | −63° (antena) | 0° |
| yaw_ref en el tick en que FIX planea | el rumbo anterior | el nuevo (1 tick de diferencia) |
| DONE | Buoy 0 (neutra) | Buoy 0 (neutra) |

Igual en los dos: estados y transiciones, estimador de deriva, plan (el
port simplifica `D·200·dist/(560 + 1e-6)` = `D·dist/(2k)`), desenrollado
del rumbo en DIVE, peculiaridad del tick en que se acaban los waypoints,
constantes restantes.

**Cableado (`diagrama_fsm.png`):**
- `Buoy_ref` → ganancia −1 → switch manual → prefiltro → limitador → TF →
  `delta_B_ref` del VBD: **`BUOY_SIGN = −1` confirmado**.
- `yaw_roll_coupling_gain` → switch (contra la constante 1) →
  `rollyaw_dir` = `turn_sign`.
- `mission_complete` → bloque `stop`: detiene la simulación 20 s después.
  El port no lo tiene: la simulación dura lo que diga `T_FINAL`.

DONE con Buoy = 0 también está en Simulink. El autor decidió (2026-10-24)
que debe quedar con flotabilidad positiva.

---

## 2026-10-25 (16) — Integral de yaw en su propio panel

En `09_control.png` (modo pitch_roll_yaw), `mem[0]` del lazo de yaw (KI·∫e,
en grados de roll, con ±`I_MAX_Y`) pasa del panel de roll a un séptimo
panel abajo. Antes quedaba aplastado por la escala del roll. Entra a la
referencia de roll como −`mem[0]`·signo_giro.

Prueba con las ganancias que el autor ajustó (`KP_Y = 0.45`, `KD_Y = 0.2`,
`KI_Y = 0.001`, `I_MAX_Y = 0.005` rad = 0.29°) y la tabla de 250 s (giro
de 0 a 45° bajando en t = 50): la integral llega al tope (−0.29°) en
t ≈ 72 s, se queda ahí hasta t ≈ 170 s y luego se descarga despacio
(−0.25° en t = 250 s).

**Sintonía final del autor** ("suficientemente bien", control a 100 Hz):
`KP_Y = 0.8`, `KD_Y = 0.2`, `KI_Y = 0.01`, `I_MAX_Y = 0.02` rad (1.15° de
roll), `ROLL_MAX = 0.8` rad; roll `KP_R = 0.008`, `KD_R = 0.02`,
`KI_R = 0.01`, `I_MAX_R = 0.001` m. Siguiente paso: waypoints.

---

## 2026-10-25 (15) — Yaw: PID del autor; modo pitch_roll_yaw; rumbo por el camino corto

Antes, el autor subió el control a 100 Hz (`PERIODO_CONTROL = 0.01`) y
sintonizó roll: KP_R 0.008, KD_R 0.02, KI_R 0.0008, I_MAX_R 1 mm;
ROLL_MAX 0.8 rad.

Ley de yaw del autor (Simulink):

```
delta_yaw = yaw - yaw_ref            (con atan2: camino corto)
roll_ref  = (-0.35*e - 0.2*ed - 0.003*ei) * rollyawdir
```

- `yaw_control`: e = wrap(ψ − ψ_ref), ė = ψ̇ − ψ̇_ref con ψ̇ de la
  cinemática (no r), aporte integral KI·∫e con tope ±`I_MAX_Y` (0.1 rad,
  provisional), roll_ref saturado en ±`ROLL_MAX`. Se quita `K_YAW`.
- **Camino corto también en el moldeado.** El limitador de razón del rumbo
  iba hacia el valor crudo: con una tabla 170° → −170° daba la vuelta
  larga (340°). Ahora el rumbo crudo se lleva a ±180° del estado del
  limitador (`make_ref_pipeline`). En la misión la FSM ya lo hacía (estado
  DIVE: rumbo relativo al actual, líneas 236-239 de V2).
- **Modo `pitch_roll_yaw`** en `02_simulacion.py` (reemplaza a `"rumbo"`):
  tabla `FASES_PITCH_ROLL_YAW` (t, pitch, rumbo, dB, nombre); el signo de
  giro sale del pitch de la tabla, igual que la FSM: bajada +1, subida −1,
  pitch 0 → 0. La figura muestra el aporte integral del yaw en el panel de
  roll.

Resultados (control a 100 Hz; escalón de rumbo 0 → 90° en t = 100 s):

| Caso | Final | Sobrepaso | ±1° desde | Aporte integral |
|---|---|---|---|---|
| Bajada (−25°) | 90.00° | 12.8° | t = 493 s | topado en −5.7° |
| Subida (+25°) | 90.00° | 12.9° | t = 492 s | topado en −5.7° |
| Bajada, KI_Y = 0 | 90.00° | 0 | t = 278 s | — |
| Subida, KI_Y = 0 | 90.00° | 0 | t = 272 s | — |

El sobrepaso viene del integrador: la referencia moldeada avanza a
0.03 rad/s (1.7 °/s) y el planeador gira más lento (≈1 °/s con 15° de
roll); el error crece y el integrador se carga. Opciones para sintonizar:
`I_MAX_Y` y `KI_Y`, o bajar la razón del moldeado de rumbo.

- Misión de waypoints (demo, control a 0.25 s): completa (5/5).
- Con `turn_sign = 0` (FSM en superficie o en el fondo) el integrador de
  yaw sigue integrando hasta su tope; al empezar la siguiente inmersión
  arranca cargado. Revisar en el modo de waypoints.

---

## 2026-10-24 (14) — Roll: ley del autor (PID no lineal + compensación de gravedad)

El autor pasó su ley de roll completa:

```
y_g = (1/cos φ)·[KP·e + KD·ė + KI·∫e − (d_φ/W)·φ̇ − z_g·sin φ − (m_φ/W)·φ̈_d + Σ_ψ]
```

Pidió implementar solo la parte de PID no lineal con compensación de
gravedad.

- `roll_control`: e = φ − φ_ref (medido − ref, como pitch),
  u = (1/cos φ)·(KP·e + KD·ė + I − z_g·sin φ), sin γ = +u·m_tot/(m_rot·R).
  Ganancias de Simulink por ahora (0.008, 0, 0.01; tope 4 mm).
- **El y_g de la ley es hacia babor** (u = −y_g del cuerpo). Lo decide el
  término −z_g·sin φ: sostener φ > 0 necesita el CG a estribor, y la ley
  da −z_g·tan φ. Así funciona en el simulador. Tomarlo como y_g de
  estribor diverge (−85°). En el diseño queda revisar el signo del término
  de y_g del modelo de roll (en Fossen g₄ lleva −y_g); ver
  [DISENO_ROLL.md](DISENO_ROLL.md), sección 8.
- `nominal_mass(params)`: m_tot y z_g0 de las masas conocidas, compartido
  por pitch y roll. Pitch queda igual (diferencia 0.0).
- Resultados con las ganancias de Simulink (escalón 15°): pico 35.6°
  (antes 23.3° con el PID simplificado). La ley cancela la rigidez del
  péndulo (W·z_g, z_g = 6.9 mm) que el PID simplificado tenía gratis, y
  con KP = 8 mm la rigidez baja a la mitad. El aporte integral final baja
  de 1.8 a 0.06 mm. Tabla de ganancias de prueba en DISENO_ROLL.md §8.
- Misión de waypoints: completa (5/5).

---

## 2026-10-24 (13) — Roll: PID simplificado del autor

El autor pasó su PID de roll de Simulink:

```
y_g_control = 0.008*e + 0*0.01*ed + 0.01*ei
gamma = asin( sat(-(yg * m_total)/(m_rot * r_rot), ±1) )
```

- `roll_control`: KP_R = 0.008, KD_R = 0 (0*0.01: derivada apagada),
  KI_R = 0.01. La integral sigue la regla de la entrada 12: se guarda el
  aporte KI·∫e (m de y_g, `mem[0]` de roll) con tope ±`I_MAX_R`
  (provisional 4 mm).
- m_total, m_rot y r_rot salen de los parámetros (58, 5, 0.07), no
  escritos a mano.
- ė = φ̇_ref − φ̇ con φ̇ de la cinemática, no p (pedido del autor).
- **Signo del error: e = φ_ref − φ**, al revés que en pitch. Con
  e = φ − φ_ref y estas ganancias el lazo diverge: roll −85°, γ en +90°.
  Razón: y_g > 0 da roll positivo y x_g > 0 da pitch negativo.
  Confirmado por el autor: en Simulink está ref − medido. En su diseño
  usa medido − ref; el tema queda planteado en
  [DISENO_ROLL.md](DISENO_ROLL.md) (con u_φ = −y_g es el mismo lazo, bit a
  bit). Por ahora se queda como en Simulink.
- Se quitó la ganancia del P provisional (`K_ROLL`).
- `02_simulacion.py`: la figura de control grafica también `mem[0]` de
  roll.

Resultados (bajada −25°, escalón de roll 0 → 15° en t = 200 s):

- Roll: sobrepaso a 23.3° (55 %), oscila con periodo ~3 s, queda en
  15.000° desde t = 207 s. Sin error en estado estacionario (antes 9.5°
  con el P).
- Aporte integral: sube hasta 3.9 mm (casi el tope de 4) durante la
  subida del roll; en equilibrio 1.8 mm ≈ z_g·tan 15°. Con KI/KP = 1.25
  s⁻¹ la integral es rápida comparada con la respuesta: buen candidato
  para la sintonización. Con `I_MAX_R` = 1 mm el roll se queda en 11.9°.
- Subida (+25°, roll −15°): simétrico (−23.3°, final −15.000°).
- Modo rumbo (0 → 45°): pasa 44° en t = 249 s (antes 264 s); roll máximo
  26.4°.

El autor cambió `KI_P` de 0.003 a 0.005 en `autopilot.py`; los resultados
de pitch de esta entrada ya lo incluyen (pico −26.2°).

---

## 2026-10-24 (12) — Pitch: θ̇ en lugar de q, derivada de la referencia, integral con tope

Observaciones del autor sobre el PID provisional de pitch:

- **q no es θ̇.** q es la velocidad angular del cuerpo; θ̇ sale de
  η̇ = J(η)ν (θ̇ = q·cos φ − r·sin φ). La parte D usaba q. Ahora usa θ̇
  (`euler_rates(x)` en `autopilot.py`). Con φ = 0 son iguales.
- `e = medido − referencia` es correcto (así está en su controlador).
- **Integral:** quitar la lógica de anti-windup condicional. Ahora se
  guarda el aporte I = KI·∫e (m de x_g) y se satura en ±`I_MAX_P`: KI
  fija la rapidez y `I_MAX_P` el máximo, para ver con claridad cuánto
  aporta la parte integral.

Duda del autor: ¿derivar el error, o θ̇ − θ̇_ref? Son lo mismo (ė =
θ̇ − θ̇_ref); lo que cambia es cómo se calcula. Se implementó θ̇ de la
cinemática (exacta, sin retraso) menos θ̇_ref como diferencia hacia atrás
de la referencia moldeada (suave). Los tres lazos reciben ahora la
derivada de su referencia (`*_ref_dot`).

Comparación, bajada a −25° con dB = −3 N desde el reposo a 200 m:

| Parte D | Pico | Asentamiento (±0.5°) | Error máx. vs ref usada | Aporte integral máx. |
|---|---|---|---|---|
| KD·(θ̇ − θ̇_ref) | −26.1° | 33 s | 3.9° | 2.6 mm |
| KD·θ̇ (solo medición) | −30.8° | 63 s | 11.3° | 5.0 mm (topado) |

Con solo θ̇, la parte D frena al planeador mientras la referencia todavía
se mueve (KD/KP = 5.3 s, y la referencia baja a 0.1 rad/s): se atrasa, el
integrador se carga y luego sobrepasa. La subida (+25°, dB = +3 N) da lo
mismo (26.2° contra 30.2°).

El aporte integral en equilibrio es ~2.0 mm de x_g bajando y −2.4 mm
subiendo (≈ 8–9 mm de rc). Con `I_MAX_P` = 1.5 mm no alcanza: el pitch se
queda en −22.8° en lugar de −25°. Con el valor provisional (5 mm) el tope
no se alcanza en estas pruebas.

También: `demos/demo_mission.py` leía la FSM en los índices viejos de
`ctrl_state` (10, 11, 20:22, 22); quedaron mal desde la entrada 11 (la
FSM empieza ahora en 24). Corregido.

---

## 2026-10-24 (11) — Un lazo por función, modos pitch_roll / rumbo, bug del periodo de control

### Controladores separados

Pedido del autor: sintonizar primero pitch y roll cada uno por su lado, y
después pasar a que la referencia de roll venga del control de yaw.

- `autopilot.py`: `attitude_step` se partió en `pitch_control`,
  `roll_control` y `yaw_control`, cada uno con su memoria de `N_MEM = 4`
  números (antes había un solo número para todo el controlador).
- `attitude_step(..., roll_from_yaw)`: la referencia de roll viene de la
  tabla (False) o del control de yaw (True).
- `make_attitude_controller(refs_fn, dt, roll_from_yaw)`: refs_fn da
  [pitch, yaw, roll, dB, signo_giro].
- `ctrl_state` nuevo: memoria 0–11, referencias usadas 12–14, moldeado
  15–23, misión 24–39.
- `02_simulacion.py`: `MODO = "directo" | "pitch_roll" | "rumbo"` con sus
  tablas; la figura de control muestra la referencia usada.
- Verificado: con el controlador provisional, modo rumbo y misión por
  waypoints dan exactamente lo mismo que antes (0.0).
- Guía de cómo implementar tus controladores:
  [CONTROL_Y_MISION.md](CONTROL_Y_MISION.md), sección 5.

### Bug: el controlador corría más rápido que PERIODO_CONTROL

`sim.py` decidía cuándo correr el controlador con el contador de pasos
**dentro de cada muestra guardada**, que se reinicia en cada muestra. Solo
era correcto si `ctrl_every` divide a `decimate`. Con `DT = 0.005`,
`MUESTREO = 0.05` y `PERIODO_CONTROL = 0.25`, el controlador corría cada
0.05 s mientras el integrador y el moldeado suponían 0.25 s: integral 5
veces más fuerte → **ciclo límite** en pitch (±60°, periodo ~28 s, masa
móvil de tope a tope).

- Arreglo: contador global de pasos.
- No afecta: lazo abierto (`ctrl_every = 1`), `run_test.py`, demos y
  misión (20 pasos por muestra con control cada 5 o 10 pasos). Verificado:
  validación y misión dan exactamente lo mismo (0.0).
- Ahora los comandos cambian cada 0.250 s exactos.
- **Los resultados del modo control de la entrada (7) se obtuvieron con el
  controlador a 0.1 s** (MUESTREO = 0.1), no a 0.25 s. Los resultados
  corregidos están en [SIMULACION.md](SIMULACION.md).

---

## 2026-10-24 (10) — Se quita disturb_xlin; error de cero equivalente

Decisión del autor: quitar `disturb_xlin`, porque la masa de perturbación
ya produce el error de cero (entrada 9). En su lugar, una función
informativa: `dynamics.disturbance_zero_equivalent(p)` da el error de cero
equivalente a `m_dist` en mm de masa móvil (`m_dist·x_dist/m_lin`) y en
grados de γ (`asin(−m_dist·y_dist/(m_rot·R))`). La imprimen
`00_balance_estatico.py` y `02_simulacion.py`.

Verificación:

- Sin perturbación, la validación da exactamente lo mismo (0.0).
- 0.5 kg en (0.3, 0.02, 0) m: equivalente +10.000 mm y −1.637°; el balance
  estático, resolviendo el modelo, da rc para pitch 0 de −10.000 mm
  respecto al nominal y γ para roll 0 de +1.637° (iguales y de signo
  contrario, como debe ser).

---

## 2026-10-24 (9) — ¿Perturbación de calibración en γ?

El autor observó que la masa de perturbación ya mete un error de cero,
porque el cero virtual se calcula sin ella, así que las perturbaciones en
la posición de los actuadores son redundantes. Comprobado: `disturb_xlin =
2 mm` y `m_dist = 0.1 kg` en x = 0.3 m (mismo momento) con `V_displaced`
compensado dan el mismo vuelo (0.004° de diferencia en pitch en 800 s);
sin compensar el volumen cambia la velocidad por el peso extra (0.670 vs
0.580 m/s bajando). Decisión: no se agrega una perturbación de
calibración en γ; `disturb_xlin` se queda (está en Simulink, en mm, sin
peso). Tabla en [MODELO.md](MODELO.md), sección 2.5.

---

## 2026-10-24 (8) — Perturbaciones: calibración y masa de perturbación; signo de giro

### Decisión del autor sobre las perturbaciones

- Se mantiene la del control: `disturb_xlin`, error en el cero virtual
  de la masa móvil (la masa no queda donde el control cree).
- Se quitan `disturb_xg`/`disturb_yg`: sumarlas a r_g después de armar M
  no es correcto (el CG se mueve sin que se mueva ninguna masa).
- Una perturbación real se modela como masa: `m_dist` en `r_dist`. No
  entra en el cálculo de los ceros ni en el control.

### Cambios

- `mass_geometry`: quinta masa puntual `m_dist` en `r_dist` (M, I_b, r_g,
  peso). No desplaza agua.
- Se quitan `disturb_xg`, `disturb_yg` y el switch
  `replicate_disturb_wiring`; M, C_RB y la restauración usan el mismo r_g.
- `00_balance_estatico.py`: cero virtual **nominal** (sin perturbaciones)
  y, aparte, el efecto de las perturbaciones.
- `02_simulacion.py`: la flotabilidad neta incluye `m_dist`; imprime las
  perturbaciones si hay.
- `cg` → `turn_sign` en el código y "signo_giro" en el script y la
  documentación (`cg` parecía "centro de gravedad"). También se renombró
  la variable local `cg` (cos γ) de `mass_geometry`.

### Verificación

- Sin perturbaciones, la validación da exactamente lo mismo que antes
  (0.0).
- 0.7 kg de perturbación puestos en la masa móvil (rc = 0) = m_lin + 0.7 kg
  (diferencia 1e-13); flotabilidad neta −0.7·g − 2 N = −8.867 N.
- Balance estático con 0.5 kg en x = 0.3 m y `disturb_xlin = 2 mm`: rc
  para pitch 0 pasa de +0.013 a −11.987 mm (10 mm por la masa + 2 mm por
  la calibración); pitch en reposo con rc = 0: −24°; V_displaced para
  neutralidad +0.5 L.

---

## 2026-10-24 (7) — Restauración verificada; modo control en la simulación manual

### Vector de restauración

El autor proporcionó su función (guardada en
`referencia_simulink/fcn_restauracion.m`): es el g(η) de Fossen. El port
calcula −g(η) con fuerzas físicas y lo suma al lado derecho. Verificado
con 2000 casos al azar: diferencia 1e-13. Con esto, todo el modelo físico
del perfil `simulink` está comparado contra Simulink. Detalle en
[MODELO.md](MODELO.md), sección 2.7.

### Cadenas de actuadores

El autor confirmó que los valores (prefiltros, limitadores, TFs) son los de
Simulink; se quitaron las marcas CONFIRMAR.

### Volumen desplazado

`V_displaced` es el volumen con el pistón a la mitad de su carrera
(Δx = 0); el VBD suma o resta ±0.503 L alrededor. Aclarado en el archivo
de configuración y en [MODELO.md](MODELO.md), sección 2.4.

### Modo control en `02_simulacion.py`

- `MODO = "directo"` (tabla de actuadores) o `"control"` (tabla de
  referencias: pitch, rumbo, dB, cg).
- El control usa los controladores **provisionales** que ya existían
  (`autopilot.py`: PID de pitch, P rumbo → roll, P roll) y el moldeado de
  referencias de Simulink. Son los que usa la misión por waypoints.
- Arreglo: el moldeado de referencias arrancaba siempre en 0 (un rumbo
  inicial de 90° con referencia 90° hacía girar al planeador). Ahora
  arranca en la actitud inicial (`make_ref_pipeline(init=...)`,
  `make_attitude_controller(init_refs=...)`). Con init en 0 es idéntico a
  antes.
- `autopilot.piecewise_schedule`: tabla por tramos compartida por el lazo
  abierto y las referencias. La validación en lazo abierto da exactamente
  lo mismo (0.0).
- Prueba: pitch_ref −25° → −25.0°; rumbo 0° → 45° en ~60 s con roll ~17°,
  con ~3° de giro inicial al lado contrario. Ver
  [SIMULACION.md](SIMULACION.md).

### disturb_xg / disturb_yg

El autor pensaba que pasan por r_lin. En su `Mbr+Ma fcn`:

```matlab
r_g=(m_s*r_s + m_lin*r_lin + m_rot*r_rot + m_p*r_p) / m;
r_g(1)=r_g(1)+dist_xg;
r_g(2)=r_g(2)+dist_yg;
```

y M se arma con `s(r_lin)`, `s(r_rot)`, etc., no con r_g. Así que dist_xg
y dist_yg solo llegan a C_RB y a la restauración. Lo único que pasa por
r_lin es `distxlin` (`fcn_r_lin`). El port hace exactamente eso. Pendiente:
saber qué señal entra a dist_xg/dist_yg en el diagrama y decidir.

---

## 2026-10-24 (6) — ρ y volumen desplazado; balance estático; simulación manual

### Flotabilidad separada de las masas (decisión final del autor)

Después de la entrada (5), el autor reconsideró: es más riguroso que la
flotabilidad no siga a las masas, para no perder el efecto de quedar fuera
de neutro; la neutralidad se ajusta a mano.

- `[flotabilidad]` nueva en el archivo de configuración: `rho = 1000`,
  `V_displaced = 0.058 m³`. `B_neutral = rho·g·V_displaced`.
- `RHO` deja de ser constante en el código: todo usa `p.rho`.
- Con los valores actuales B − W = 0 exacto; la prueba de validación da
  lo mismo que antes (diferencia 6e-14 m, redondeo).

### `sim.py`

- Solvers de paso fijo con nombres de Simulink: `ode1` (Euler), `ode2`
  (Heun), `ode4` (RK4, default). `make_simulator(..., solver="ode4")`.
- `actuator_steady_state(p, rc, gamma, dB)`: estados de los actuadores ya
  asentados, para arrancar con los actuadores en posición. Verificado: rc,
  γ y la flotabilidad neta quedan constantes desde t = 0.
- `build_initial_state(...)`: posición y actitud en el inercial,
  velocidad en el cuerpo o en el inercial (se convierte), velocidad
  angular, actuadores. Verificado: NED → cuerpo → NED regresa igual.

### `escenarios/00_balance_estatico.py`

Masas, flotabilidad neta, V_displaced para neutralidad, CG/CB/BG, cero
virtual de la masa móvil (pitch 0 en reposo), γ para roll 0, equilibrio en
reposo y curvas. Resultados en [SIMULACION.md](SIMULACION.md), sección 1:
cero virtual calculado 0.28611 m (archivo: 0.2861), BG = 6.9 mm,
sensibilidad ~1°/mm.

### `escenarios/02_simulacion.py`

Simulación por tiempo para probar a mano: solver, paso (0.005 s), duración,
condiciones iniciales, tabla de actuadores. Guarda la corrida, mueve la
anterior a `anterior/`, imprime qué cambió (entradas, parámetros y
switches) y superpone las dos. Probado con tres corridas seguidas (cambio
de rc; cambio a perfil `completo` con velocidad inicial en el inercial).
Detalle en [SIMULACION.md](SIMULACION.md), sección 2.

### Experimento: paso de integración

| Comparación (pitch, prueba de validación) | Máxima | Estacionario |
|---|---|---|
| dt 0.05 vs 0.005 | 0.54° en t = 608 s (cambio bajada → subida) | 0 |
| dt 0.005 vs 0.0025 | 0.03° | 0 |
| ode1 / ode2 / ode4 con dt 0.005 | < 0.002° entre ellos | |

---

## 2026-10-24 (5) — Flotabilidad neutra sigue a las masas; MA diagonal

### Decisiones del autor

- **La flotabilidad neutra debe seguir a las masas.** Se van a hacer
  experimentos con distintos parámetros. Se revirtió el cambio del commit
  anterior (`m_displaced` fijo en 58 kg, como el `W_total` de Simulink):
  otra vez `B_neutral = (m_static + m_lin + m_rot + m_pist)·g`.
- **Masa añadida: acoplamientos en 0**, solo la diagonal.
- **Superficie:** confirmado, `B_scale` multiplica a B y `Δx_b` se suma
  a x_b.
- **`disturb_xlin`** representa un error en el cero virtual de la masa
  móvil (rc no queda donde el control cree). Entra en todo: M, I_b, r_g,
  restauración. Así está en el port.
- `disturb_xg`/`disturb_yg` (entran en r_g después de M): pendiente, el
  autor lo está pensando.

### Verificación

- Con la MA anterior, el código nuevo da exactamente lo mismo que el commit
  anterior (0.0): con las masas actuales, la suma es 58 kg.
- MA diagonal vs la anterior: 1.2 cm horizontal en 1200 s; la tabla por
  fase no cambia en ningún decimal.
- B − W = 0 con dB = 0 para m_lin = 18, m_static = 40, y m_rot = 7 con
  m_pist = 1.5.

### Experimento: m_lin = 18 kg con los mismos comandos

El CG se va adelante: bajada a −72° (vs −38°). En la subida (rc = −0.03)
la masa no alcanza a levantar la nariz: queda a −49° con α ≈ −100° y sube
a 0.05 m/s. Los comandos en lazo abierto están ajustados al vehículo base;
con otras masas hay que volver a ajustar rc (en lazo cerrado lo haría el
control). Además ese α está fuera del rango donde vale el modelo en α y β.

---

## 2026-10-24 (4) — Decisiones del autor, últimas funciones y archivo de configuración

El autor proporcionó `fcn_VBD_actuator`, `fcn_buoyancy_surface` y la
función que arma `r_lin`. Guardadas en `referencia_simulink/`. Comparación
en [MODELO.md](MODELO.md), secciones 2.4 a 2.6.

### Decisiones del autor

- **No replicar errores de Simulink.** La inercia de la masa rotatoria
  queda como `R·I·Rᵀ`; se quitó el switch `rot_inertia_mode`.
- **Masa añadida con simetría babor–estribor**: se conservan tus 0.1 (u–w)
  y 0.01 (p–r), y u–v, v–w, p–q, q–r van en cero.
- **Sin pitch automático en el VBD** (`p_ref = ±con`): el pitch de cada
  inmersión se da explícitamente.
- **Parámetros en un archivo de configuración**, no en el script.
- **"dist" → "disturb"** en los nombres, para no confundir con distancia.

### Commit 1: modelo

- `mass_geometry`: `R·I·Rᵀ` fijo; `r_lin` incluye `disturb_xlin`.
- `dist_xg`, `dist_yg` → `disturb_xg`, `disturb_yg`; nuevo `disturb_xlin`;
  `replicate_dist_wiring` → `replicate_disturb_wiring`.
- `use_surface_model` (bool) → `surface_model`: `"none"`, `"simulink"`
  (port exacto de `fcn_buoyancy_surface`), `"full"` (el del port).
- MA por defecto: la simétrica.

Verificación: con MA diagonal y superficie `"none"`, el código nuevo da
exactamente lo mismo que el commit anterior (diferencia 0.0). Con
`"simulink"` lejos de la superficie, también 0.0 contra `"none"`.

### Commit 2: archivo de configuración

- `config/cidesi.toml`: todos los valores numéricos, con unidades y origen,
  más los perfiles de switches `[modelo.simulink]` y `[modelo.completo]`.
- `params.py`: `load_config(archivo, modelo, dt)`, `load_params`,
  `load_model`. Revisa que no falte ni sobre ninguna clave y que las
  matrices de masa añadida sean simétricas. `cidesi_defaults()` ahora lee
  el archivo (los scripts de v1 siguen funcionando).
- VBD como en Simulink: `B_neutral = m_displaced·g` (constante, 58 kg) en
  lugar de la suma de las masas; área con π exacto (`vbd_diam = 0.08`).
- El script de validación usa `load_config` y guarda una copia del archivo
  usado en `out/01_validacion_dinamica/config_usada.toml`.

Verificación: todos los parámetros iguales al commit 1, salvo el área del
pistón (diferencia relativa 1e-9 por π); diferencia en la simulación de
4.5e-8 m. Un error de dedo (`KDO`) o una clave faltante dan error con el
nombre. `run_test.py` sigue terminando a 139.9 m.

### Experimentos

| Prueba | Resultado |
|---|---|
| MA simétrica vs diagonal (validación, 1200 s) | 1.2 cm horizontal, 0.01° yaw |
| MA de Simulink completa vs diagonal | 0.38 m horizontal, 0.13° yaw |
| `disturb_xlin = 5 mm` | pitch de bajada −38.5° → −44.5°, de subida +36.6° → +29.1° |
| subida a la superficie, `"simulink"` vs `"full"` | ambos flotan a z ≈ 0.13 m con B_scale 0.995; pitch +2.1° vs +3.2° |

La tabla por fase del script de validación no cambia en ningún decimal.

---

## 2026-10-24 (3) — Revisión de `Mbr+Ma fcn` y Coriolis

El autor proporcionó las dos funciones. Guardadas tal cual en
`referencia_simulink/fcn_Mbr_Ma.m` y
`referencia_simulink/calculate_coriolis_total_fossen.m`. Comparación línea
por línea en [MODELO.md](MODELO.md), secciones 2.2 y 2.3.

### Resultado

- **Igual:** masas, posiciones, tensores, M_RB, I_b, r_g, C_RB, el
  cableado de la perturbación de CG (`replicate_dist_wiring`) y C_A anulado
  (`C_A*0`).
- **Diferente 1:** rotación del tensor de la masa rotatoria. Simulink:
  Rᵀ·I·R. Port: R·I·Rᵀ, que es el consistente con `r_rot`. Solo cambia el
  signo de un término cruzado yz de ~0.001 kg·m².
- **Diferente 2:** Simulink tiene 0.1 kg (MA) y 0.01 kg·m² (IA) fuera de la
  diagonal de la masa añadida; el port los había puesto en cero.

### Cambios en la librería

- `ModelConfig.rot_inertia_mode`: `"RIRt"` (default, comportamiento
  anterior) o `"RtIR"` (= Simulink).
- `params.MA_SIMULINK`: masa añadida completa de tu función. Se usa con
  `p._replace(MA=MA_SIMULINK)`. El default sigue siendo diagonal.
- El script de validación ahora usa las dos opciones de Simulink.

Defaults sin cambio: verificado bit a bit (diferencia 0.0) contra la
telemetría guardada antes del cambio.

### Efecto medido (escenario de validación, 1200 s)

| Variante vs port por defecto | Horizontal | Profundidad | Yaw |
|---|---|---|---|
| solo `RtIR` | 0.0003 m | < 0.0001 m | 0.0009° |
| solo `MA_SIMULINK` | 0.38 m | 0.0007 m | 0.13° |

La tabla de valores por fase del script no cambia en ningún decimal: con
ν̇ = 0 la matriz de masa no influye, las diferencias vienen solo de los
transitorios.

---

## 2026-10-24 (2) — Librería + script de validación en lazo abierto

Objetivo: tener la librería con toda la dinámica y un script para validarla
en lazo abierto, sin geolocalización, graficando lo pertinente.

### Git

Repositorio local en `sim/gliderjax/` (autor `cdelacruz`,
cdelacruz.ctrl@gmail.com, configurado solo para este repositorio).
`.gitignore`: `out/`, `*.npz`, `__pycache__/`.

| Commit | Contenido |
|---|---|
| v1: copia exacta | punto de partida = `_old/gliderjax_WP_2026-07` |
| docs | documentación inicial |
| dynamics | saturación como switch, línea muerta, `B_net` |
| scope + autopilot | vistas y análisis nuevos, controlador de lazo abierto |
| escenarios + docs | script de validación, esta entrada |

### Cambios en la librería

- `ModelConfig.use_alpha_saturation` (default False). **Cambia el
  comportamiento por defecto**: los scripts de v1 (`run_test.py`, demos)
  ahora corren sin saturación. `run_test.py` termina a 139.9 m en lugar
  de 139.8 m.
- Se quitó una línea muerta en `dynamics.py` (`x_b_eff` se calculaba dos
  veces y la primera se descartaba).
- Telemetría nueva `B_net` = flotabilidad neta [N].
- `autopilot.make_open_loop_controller(tabla)`: comandos constantes por
  tramos.
- `scope.py`:
  - figuras: `overview` (resumen de una página), `actuators_plot`,
    `flow_frame_wrench` (= Fh, Th de `fcn_damping.m`), `trajectory_xy`,
    `trajectory_3d`;
  - `compare` para superponer corridas, `phase_summary` para los valores
    por fase;
  - `save_tel` / `load_tel` y `from_simulink_csv`;
  - `moments_breakdown` ahora incluye superficie y crossflow (antes los
    omitía) y la suma de todos los términos;
  - ya no fuerza el backend `Agg`: con `scope.KEEP_OPEN = True` +
    `plt.show()` las figuras se abren con zoom y pan;
  - `yaw_deg` se desenvuelve, sin saltos de ±180°.

### Verificación (diferencia máxima en η, ν_r, α, β, V)

| Comparación | Resultado |
|---|---|
| nueva con saturación ON (resto defaults) vs v1 WP | 0.0 |
| nueva con saturación ON + superficie/crossflow/geodesia OFF vs v1 NO_WP | 0.0 |
| nueva con saturación OFF vs experimento sin saturación | 0.0 |

### Primera corrida del script de validación

Ver tabla de resultados en [VALIDACION.md](VALIDACION.md). Planeo recto:
pitch ∓38.5°/+36.6°, α ±1.34°, V ≈ 0.56 m/s, razón de planeo ≈ 1.2.
Con γ = +30°: roll −26°, bajando gira a la izquierda y subiendo a la
derecha.

---

## 2026-10-24 (1) — Reorganización y diagnóstico

### Decisiones

- **Sin GUI por ahora.** El código todavía cambia (controladores
  provisionales, formación de N planeadores en el futuro). Se prefiere
  ordenar el código y documentar dónde se cambia cada cosa.
- **Una sola librería** con distintos scripts por caso de uso (validación sin
  waypoints, misión con waypoints, después formación), en lugar de copias
  separadas.
- **Construir paso a paso, documentando todo** en `docs/`.

### Archivar v1 y crear la versión de trabajo

- `gliderjax/` → `_old/gliderjax_WP_2026-07/`
- `gliderjax_NO_WP/` → `_old/gliderjax_NO_WP_2026-07/`
- Nueva `gliderjax/` = copia exacta de la WP, sin `out/` ni `__pycache__`.
  Verificado con `diff -r`.
- Ambas versiones viejas corren desde `_old/` (probado con su `run_test.py`):
  WP termina a 139.8 m; NO_WP a 93,667 m (su `N_FRAMES = 259000`).
- No se renombró `sim/` porque contiene el venv.

### Hallazgo: la versión WP contiene a la NO_WP

La WP es la NO_WP más tres modelos con switch (`use_surface_model`,
`use_crossflow`, `use_geodesy`). Prueba: escenario de `run_test.py`
(rc = 0.03 m, γ = 0, dB = −3 N, arranque en superficie con u = 0.2 m/s y
pitch −0.2 rad, 400 s) en ambas copias:

| Comparación | Diferencia máx. en η, ν_r, α, V |
|---|---|
| NO_WP vs WP con los 3 switches en False | **0.0** (bit a bit) |
| NO_WP vs WP con switches por defecto | 1.9 m en profundidad, 0.23 rad en α |

### Revisión de la función de damping de Simulink

El autor proporcionó el código (guardado en
`referencia_simulink/fcn_damping.m`). Comparado línea por línea: el port es
idéntico **salvo la saturación tanh de α y β**, que no existe en Simulink.
Detalle en [MODELO.md](MODELO.md), sección 2.1.

### Experimento: efecto de la saturación tanh

Copia temporal de la librería con `alpha_e = alpha`, `beta_e = beta`.
Resultados en [MODELO.md](MODELO.md), sección 3. Resumen: despreciable en
planeo sumergido (α ≈ 1.3°); solo cambia algo cerca de la superficie; sin
saturación no hay inestabilidad en los casos probados.

### Experimento: inversión del giro

γ = +30° fijo, bajada 400 s (rc = +0.03, dB = −3) y subida 400 s
(rc = −0.03, dB = +3), modelo Simulink, `kmr_mode` signum vs const.
Resultado: bajando gira a la izquierda y subiendo a la derecha en **ambos**
modos; sign(α) solo cambia la tasa de giro en subida (+45 %). Detalle en
[MODELO.md](MODELO.md), sección 6.

### Observaciones en la misión triángulo (6000 s, modelo completo)

- Misión completa a t = 1854 s.
- En DONE el planeador queda con flotabilidad neutra y nariz abajo (−63°),
  hundiéndose despacio. Según el autor debería tener flotabilidad positiva.
- El flare de CLIMB no tiene efecto (`FLARE_DEPTH = SURF_EPS`).

Ver [CONTROL_Y_MISION.md](CONTROL_Y_MISION.md), sección 4.
