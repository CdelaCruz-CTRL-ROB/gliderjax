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
- [ ] Comparar contra tus funciones `Mbr+Ma fcn` (inercia de la masa
      rotatoria: R·I·Rᵀ vs Rᵀ·I·R), `fcn_VBD_actuator` y
      `fcn_buoyancy_surface`.
- [ ] Confirmar `BUOY_SIGN` y los demás valores marcados `CONFIRM`.
- [ ] Revisar si sign(α) en KMR es necesario (la inversión del giro ocurre
      sin él).
- [ ] DONE con flotabilidad positiva (confirmar contra V2).
- [ ] Revisar `FLARE_DEPTH` / `SURF_EPS` contra V2.
- [ ] Reemplazar controladores provisionales por los bloques de Simulink
      (`Pitch_control`, `Yaw_control`, `Roll control`).
- [ ] Script de misión por waypoints (`escenarios/02_...`).
- [ ] Revisar los defaults de `ModelConfig` (hoy superficie, crossflow y
      geodesia vienen encendidos).
- [ ] Cheat sheet: "quiero cambiar X → dónde".
- [ ] Futuro: formación de N planeadores.

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
Detalle en [MODELO.md](MODELO.md), sección 2.

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
