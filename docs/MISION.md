# Misión por waypoints: `escenarios/03_mision.py`

Última actualización: 2026-10-25

Mismo flujo que `02_simulacion.py`: cambias valores arriba del script (o en
`config/cidesi.toml`), corres, ves las gráficas. Desde
`GLIDER_JAX\sim\gliderjax`:

```
..\Scripts\python.exe escenarios\03_mision.py
```

Resultados en `out/03_mision/`; la corrida anterior pasa a `anterior/`.

## Secciones del script

| Sección | Qué se cambia |
|---|---|
| 1) MODELO | perfil de switches (`simulink` / `completo`). La geodesia se enciende sola: la misión necesita lat/lon |
| 2) SOLVER | solver, `DT`, `MUESTREO` (default 1 s: las misiones son largas) |
| 3) DURACIÓN | `DURACION = "mision"` o `"tiempo"` (ver abajo) |
| 4) MISIÓN | `WAYPOINTS`, `FSM`, `MODELO_INMERSION`, `FSM_AJUSTES`, `PERIODO_CONTROL` |
| 5) CONDICIONES INICIALES | posición respecto al WP1 y actitud (velocidad 0, actuadores en neutro) |
| 6) SALIDA | figuras, aviso de alpha, `MOSTRAR` |

## Duración

- **`"mision"`**: corre hasta que la FSM llega a DONE, y `T_DESPUES_DONE`
  más (20 s, como el bloque `stop` de Simulink). Si no termina antes de
  `T_MAX`, se detiene y avisa **TIMEOUT** con los waypoints alcanzados.
- **`"tiempo"`**: corre `T_FINAL` y se corta, terminen o no los waypoints.

Para revisar si terminó, simula por bloques de 500 s (`BLOQUE`, sección 7)
que siguen donde quedó el anterior (`sim.make_resumable_simulator`).
Verificado: da exactamente lo mismo que una corrida de un solo bloque
(diferencia 0.0). En consola va imprimiendo t, estado de la FSM, waypoint
objetivo y profundidad.

El resultado queda en `resultado.json`: `completa`, `timeout`, `tiempo`
(se acabó T_FINAL sin terminar) o `nan`.

## Waypoints

```python
WAYPOINTS = [
    # --- origen: SIEMPRE la primera fila ---
    (-95.72275,    19.37214),     # WP0: origen

    # --- triángulo (el de Simulink) ---
    (-95.72011,    19.37464,    -25.0,  25.0),   # WP1, inmersiones a ∓25°
    (-95.72275,    19.37714),     # WP2: el pitch de la FSM (∓35°)
    # --- ... y para el cuadro de 7 puntos, además: ---
    # (-95.71798,    19.37714),   # WP3 (~500 m al este del WP2)
    # ...
    # --- cuadro de 3 km ---
    # (-95.72275,    19.40014),   # WP A: 3 km al norte del origen
    # ...
]
```

- **Trayectorias:** en el script están las de tu lista de Simulink (las del
  triángulo, cuadro de 7 puntos, un punto al norte, cuadro de 3 km y
  barrido), como bloques que se comentan o descomentan. La primera fila
  (origen) se deja siempre.
- Orden **(lon, lat)**, el de Fossen y de tu Simulink. Google Maps da
  (lat, lon). Si se invierten, en el Golfo la "latitud" queda en −95° y el
  script da error diciendo qué waypoint es.
- **Numeración como en tu lista:** el origen es el **WP0**, y los objetivos
  son WP1, WP2, … Así se muestran en consola y en las figuras. La FSM, por
  dentro, numera desde 1 como en Simulink (`current_wp_index`: 1 = WP0).
- El WP0 es el origen del marco NED y el punto de arranque. En el primer
  FIX la FSM lo da por alcanzado (distancia 0) y va al WP1. "Waypoints
  alcanzados" cuenta solo los objetivos (sin el origen).
- **Pitch por tramo** (opcional): el de todas las inmersiones **hacia** ese
  waypoint, bajada < 0 y subida > 0. Sin él se usa el de la FSM
  (`pitch_down_rad`, `pitch_up_rad`). Por inmersión no se puede: la FSM
  decide cuántas inmersiones hace en cada tramo. El flare arranca del pitch
  de subida del tramo.

`FSM_AJUSTES` cambia constantes de la FSM, por ejemplo
`{"acceptance_radius_m": 50.0}`. La lista está en
[CONTROL_Y_MISION.md](CONTROL_Y_MISION.md), sección 3.2. `FSM` y
`MODELO_INMERSION` eligen por nombre una FSM de `gliderjax/fsm/` y un
modelo de `fsm/dive_models.py` (sección 3.1).

## Salida

En consola:
- la tabla de waypoints (N, E, longitud del tramo, pitch);
- el avance por bloques;
- el resultado;
- qué cambió respecto a la corrida anterior, y su resultado;
- **una fila por inmersión**: inicio, waypoint objetivo, rumbo y
  profundidad planeados, profundidad máxima real, duración, y distancia al
  waypoint al salir a superficie.

| Figura | Archivo | Contenido |
|---|---|---|
| `mapa` | `10_mapa.png` | vista superior con el formato de tu figura de Simulink (ver abajo) |
| `perfil` | `11_perfil.png` | profundidad contra la planeada, estado de la FSM, waypoint objetivo |
| `actitud` | `12_actitud.png` | pitch y rumbo (real, referencia usada, salida de la FSM), roll contra la referencia que da el control de yaw, comando dB y flotabilidad neta, signo de giro |
| `deriva` | `13_deriva.png` | D_norm estimado y distancia al waypoint en cada fix |
| `resumen`, `actuadores`, `trayectoria_3d` | `01_`, `02_`, `08_` | los mismos de `02_simulacion.py` |

El aviso de |alpha| > 15° solo cuenta el tiempo bajo el agua (z > 5 m): en
superficie, con velocidad ~0, alpha no significa nada.

### El mapa (`10_mapa.png`)

Mismo formato y simbología que tu figura "Navegación por Waypoints Bajo
Perturbaciones" (`referencia_simulink/figura_waypoints.png`):
- **Norte X** en el eje horizontal y **Este Y** en el vertical;
- letra con serifas;
- leyenda fuera del área de la gráfica, para que no tape nada.

| Símbolo | Qué es |
|---|---|
| línea roja | trayectoria |
| flechas azul claro | corriente marina, `current_n` de `[perturbaciones]` en el TOML. Es uniforme, así que las flechas son iguales y solo indican la dirección; la magnitud va en la leyenda. Sin corriente no se dibujan |
| X roja + círculo rojo punteado | waypoint objetivo y su radio de aceptación; los que se repiten (p. ej. WP2 y WP6 del cuadro de 7) llevan una sola etiqueta |
| ▼ gris oscuro | punto de inflexión: el punto más hondo de cada inmersión, con la profundidad que alcanzó |
| ▲ cian | punto de superficie: donde sale (entrada a SURFACE_WAIT) |
| ★ verde | inicio |

Es una sola corrida. No hay comparación con estimador y sin estimador:
solo se usa el formato.

## Pruebas (2026-10-25, triángulo de tu Simulink, perfil `simulink`, control a 100 Hz)

| Caso | Resultado |
|---|---|
| `"mision"` | completa: DONE en t = 1957 s; simula hasta 1977 s (14 s de cómputo). 2 inmersiones: llega a 18 m del WP1 y a 51 m del WP2 |
| `"mision"`, `T_MAX = 1200` | TIMEOUT: 1 de 2 waypoints; la 2a inmersión "no salió" |
| `"tiempo"`, `T_FINAL = 1200` | tiempo cumplido sin terminar: 1 de 2 |
| WP1 con ∓25° | completa en t = 3456 s, pero con 3 inmersiones: la 1a se pasa del WP1 por 147 m y hace una de 69 m para regresar |
| lat/lon invertidas en el WP1 | error con el WP y la fila |
| cuadro de 7 puntos con corriente de 0.09 m/s (N −0.05, E +0.08; solo en la prueba, el TOML sigue sin corriente) | completa, 6/6, DONE en t = 7685 s (54 s de cómputo), 8 inmersiones. Repite dos inmersiones cortas (29 y 46 m) para entrar al radio del WP3 y del WP6 |

(Las pruebas de la entrada 20 de la BITACORA usan la numeración vieja,
WP1 = origen.)

El caso de ∓25° muestra un límite del modelo de inmersión recto. Supone
razón de planeo k = 1.4 sea cual sea el pitch, y a −25° el planeador
avanza más (≈1.9, ver `02_simulacion.py`). Un modelo de inmersión que tome
k del pitch del tramo lo corregiría (pendiente en la BITACORA).
