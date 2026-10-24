# Bitácora

Registro de pasos, experimentos y decisiones. Lo más reciente arriba.

---

## 2026-10-24 — Reorganización y diagnóstico

### Decisiones

- **Sin GUI por ahora.** El código todavía cambia (controladores
  provisionales, formación de N planeadores en el futuro). Se prefiere
  ordenar el código y documentar dónde se cambia cada cosa.
- **Una sola librería** con distintos scripts por caso de uso (validación sin
  waypoints, misión con waypoints, después formación), en lugar de copias
  separadas.
- **Construir paso a paso, documentando todo** en `docs/`.

### Paso 1: archivar v1 y crear la versión de trabajo

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

Los switches vienen en True por defecto: el `ModelConfig()` por defecto
**no** es el modelo de Simulink.

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

### Pendientes

- [ ] Saturación tanh como switch, apagada por defecto (= Simulink).
- [ ] DONE con flotabilidad positiva (confirmar contra V2).
- [ ] Revisar `FLARE_DEPTH` / `SURF_EPS` contra V2.
- [ ] Confirmar `BUOY_SIGN` y los demás valores marcados `CONFIRM`.
- [ ] Reemplazar controladores provisionales por los bloques de Simulink
      (`Pitch_control`, `Yaw_control`, `Roll control`).
- [ ] Revisar si sign(α) en KMR es necesario (la inversión del giro ocurre
      sin él).
- [ ] Scripts de escenarios: validación (modelo Simulink) y misión WP.
- [ ] `scope.py`: guardar/cargar telemetría, comparar corridas, gráficas de
      misión.
- [ ] Cheat sheet: "quiero cambiar X → dónde".
- [ ] Control de versiones con git.
- [ ] Futuro: formación de N planeadores.
