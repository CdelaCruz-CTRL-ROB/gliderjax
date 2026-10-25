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
- [x] Perturbaciones: se quitan `disturb_xg`/`disturb_yg`; quedan
      `disturb_xlin` (calibración) y la masa de perturbación `m_dist`.
- [x] `cg` → `turn_sign` / "signo de giro".
- [x] Error de calibración en γ: no se agrega; una `m_dist` lateral da el
      mismo efecto (ver [MODELO.md](MODELO.md), sección 2.5).
- [ ] Barridos de parámetros: correr varias variantes y superponerlas
      ("mover todo y ver qué da"). Puede salir de `02_simulacion.py`.
- [ ] ¿Escalar los coeficientes hidrodinámicos con ρ/ρ_CFD al cambiar
      `rho`? Hoy no se escalan.
- [ ] Confirmar `BUOY_SIGN` (convención de la FSM V2 → VBD).
- [ ] Pasar al archivo de configuración las ganancias de control y el
      moldeado de referencias (cuando lleguen tus controladores).
- [ ] Script de waypoints: pitch de cada inmersión como dato de la misión
      (sin pitch automático en el VBD).
- [ ] Revisar si sign(α) en KMR es necesario (la inversión del giro ocurre
      sin él).
- [ ] DONE con flotabilidad positiva (confirmar contra V2).
- [ ] Revisar `FLARE_DEPTH` / `SURF_EPS` contra V2.
- [ ] Reemplazar controladores provisionales por los bloques de Simulink
      (`Pitch_control`, `Yaw_control`, `Roll control`).
- [ ] Script de misión por waypoints (`escenarios/03_...`).
- [ ] Cheat sheet: "quiero cambiar X → dónde" (primera versión en
      [README.md](README.md)).
- [ ] Futuro: formación de N planeadores.

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
