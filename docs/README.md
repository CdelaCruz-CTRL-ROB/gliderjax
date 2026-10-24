# Documentación del simulador gliderjax

Simulador 6-DOF del planeador submarino CIDESI–UNINDETEC en JAX, portado
desde el modelo Simulink. Esta carpeta es la referencia para recordar **qué
hay en el modelo, de dónde viene cada parte y dónde se cambia cada cosa**.

> El `README.md` de la raíz (en inglés) es el de la versión v1 que escribió
> la sesión anterior. La documentación vigente es esta carpeta.

## Documentos

| Archivo | Contenido |
|---|---|
| [VALIDACION.md](VALIDACION.md) | Script de validación en lazo abierto: cómo usarlo, qué muestra cada figura, cómo comparar con Simulink, señales disponibles |
| [MODELO.md](MODELO.md) | Inventario del modelo físico: qué viene de Simulink y qué se agregó, comparación con tu código, saturación, crossflow, superficie, inversión del giro |
| [CONTROL_Y_MISION.md](CONTROL_Y_MISION.md) | Controladores actuales (provisionales), FSM de waypoints estado por estado, problemas conocidos |
| [BITACORA.md](BITACORA.md) | Registro fechado de cada paso, experimento y decisión |
| [referencia_simulink/](referencia_simulink/) | Código original de Simulink (la "verdad" para validar) |

## Estructura de carpetas

```
GLIDER_JAX/sim/                       venv de Python (Lib/, Scripts/) -- NO renombrar ni mover
  _old/
    gliderjax_WP_2026-07/             versión v1 con waypoints, intacta
    gliderjax_NO_WP_2026-07/          versión v1 sin waypoints, intacta
  gliderjax/                          VERSIÓN DE TRABAJO (esta)
    gliderjax/                        la librería: física, simulador, control, navegación, gráficas
      params.py                         parámetros (GliderParams) y switches del modelo (ModelConfig)
      dynamics.py                       ecuaciones de movimiento e hidrodinámica
      sim.py                            integrador RK4, actuadores, simulador
      autopilot.py                      controladores de actitud (PROVISIONALES)
      navigation.py                     FSM de waypoints (V2) y GPS
      geodesy.py                        conversiones lat/lon <-> ECEF <-> NED
      scope.py                          gráficas tipo "scope" de Simulink
    escenarios/                       SCRIPTS DE TRABAJO, uno por caso de uso
      01_validacion_dinamica.py         lazo abierto, modelo = Simulink
    out/                              resultados (figuras, telemetría); no va a git
    run_test.py                       script de prueba de v1 (se conserva)
    demos/                            ejemplos de v1 (lazo abierto, lazo cerrado, misión, Monte Carlo)
    validation/                       comparación contra CSV exportados de Simulink
    docs/                             esta documentación
```

La librería (`gliderjax/gliderjax/`) es el equivalente de tu `.slx`; los
scripts (`run_test.py`, `demos/`) son el equivalente de los `.m` que fijan
parámetros y llaman a la simulación.

## Cómo correr

Desde `GLIDER_JAX\sim\gliderjax`:

```
..\Scripts\python.exe escenarios\01_validacion_dinamica.py
..\Scripts\python.exe demos\demo_mission.py
```

### Versiones viejas

Siguen funcionando desde `_old/` (verificado 2026-10-24). Ojo: correrlas
sobrescribe las figuras de su `out/`.

```
cd ..\_old\gliderjax_NO_WP_2026-07
..\..\Scripts\python.exe run_test.py
```

El `run_test.py` de NO_WP tiene tus últimas ediciones: `N_FRAMES = 259000`
(≈72 h de bajada sin fondo, termina a ~93.7 km) y
`replicate_dist_wiring=False`.

## Forma de trabajo

Se construye paso a paso. Cada cambio al código o hallazgo se registra en
[BITACORA.md](BITACORA.md) con fecha, y se actualiza el documento que
corresponda.

### Historial con git

`gliderjax/` es un repositorio git local (sin cuenta ni GitHub). Cada paso
es un commit. Desde `GLIDER_JAX\sim\gliderjax`:

```
git log --oneline              lista de pasos
git show <id>                  qué cambió en un paso
git diff                       cambios aún sin guardar en un commit
git diff <id> -- gliderjax     comparar la librería con un paso anterior
```

El primer commit es la copia exacta de v1, así que cualquier diferencia con
la versión vieja se ve con `git diff <id_del_primer_commit>`.
