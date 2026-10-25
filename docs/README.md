# Documentación del simulador gliderjax

Simulador 6-DOF del planeador submarino CIDESI–UNINDETEC en JAX, portado
desde el modelo Simulink. Esta carpeta es la referencia para recordar **qué
hay en el modelo, de dónde viene cada parte y dónde se cambia cada cosa**.

> El `README.md` de la raíz (en inglés) es el de la versión v1 que escribió
> la sesión anterior. La documentación vigente es esta carpeta.

## Documentos

| Archivo | Contenido |
|---|---|
| [SIMULACION.md](SIMULACION.md) | Balance estático (cero virtual, neutralidad) y simulación manual: solver, condiciones iniciales, comparación con la corrida anterior |
| [VALIDACION.md](VALIDACION.md) | Script de validación en lazo abierto: cómo usarlo, qué muestra cada figura, cómo comparar con Simulink, señales disponibles |
| [MODELO.md](MODELO.md) | Inventario del modelo físico: qué viene de Simulink y qué se agregó, comparación con tu código, saturación, crossflow, superficie, inversión del giro |
| [CONTROL_Y_MISION.md](CONTROL_Y_MISION.md) | Controladores actuales, FSM de waypoints estado por estado, problemas conocidos |
| [DISENO_ROLL.md](DISENO_ROLL.md) | Pendiente de diseño: signo del error y variable de control del roll (para trabajarlo en otra sesión) |
| [BITACORA.md](BITACORA.md) | Registro fechado de cada paso, experimento y decisión |
| [referencia_simulink/](referencia_simulink/) | Código original de Simulink (la "verdad" para validar) |

## Estructura de carpetas

```
GLIDER_JAX/sim/                       venv de Python (Lib/, Scripts/) -- NO renombrar ni mover
  _old/
    gliderjax_WP_2026-07/             versión v1 con waypoints, intacta
    gliderjax_NO_WP_2026-07/          versión v1 sin waypoints, intacta
  gliderjax/                          VERSIÓN DE TRABAJO (esta)
    config/
      cidesi.toml                     TODOS LOS PARÁMETROS y perfiles de switches (editar aquí)
    gliderjax/                        la librería: física, simulador, control, navegación, gráficas
      params.py                         estructura de parámetros y switches; lee config/*.toml
      dynamics.py                       ecuaciones de movimiento e hidrodinámica
      sim.py                            solvers (ode1, ode2, ode4), actuadores, simulador
      autopilot.py                      controladores: un lazo por función (PROVISIONALES)
      navigation.py                     FSM de waypoints (V2) y GPS
      geodesy.py                        conversiones lat/lon <-> ECEF <-> NED
      scope.py                          gráficas tipo "scope" de Simulink
    escenarios/                       SCRIPTS DE TRABAJO, uno por caso de uso
      00_balance_estatico.py            sin simular: masas, flotabilidad, CG, cero virtual
      01_validacion_dinamica.py         prueba fija en lazo abierto, modelo = Simulink
      02_simulacion.py                  simulación manual: cambias, corres, comparas
    out/                              resultados (figuras, telemetría); no va a git
    run_test.py                       script de prueba de v1 (se conserva)
    demos/                            ejemplos de v1 (lazo abierto, lazo cerrado, misión, Monte Carlo)
    validation/                       comparación contra CSV exportados de Simulink
    docs/                             esta documentación
```

La librería (`gliderjax/gliderjax/`) es el equivalente de tu `.slx`;
`config/cidesi.toml` es el equivalente del `.m` de parámetros; los scripts
de `escenarios/` definen la prueba (comandos, condiciones iniciales,
gráficas) y llaman a la simulación.

## Dónde cambiar qué

| Quiero cambiar… | Dónde |
|---|---|
| masas, inercias, masa añadida, coeficientes hidrodinámicos, amortiguamiento | `config/cidesi.toml` |
| actuadores (FT, limitadores de tasa, saturaciones) | `config/cidesi.toml`, `[actuadores]` y `[masas]` |
| volumen desplazado (neutralidad) y densidad del agua | `config/cidesi.toml`, `[flotabilidad]` (no sigue a las masas: revisar con `00_balance_estatico.py`) |
| VBD (diámetro, carrera, estación del pistón) | `config/cidesi.toml`, `[vbd]` |
| masa de perturbación y corriente | `config/cidesi.toml`, `[perturbaciones]` |
| qué partes del modelo se usan (switches) | `config/cidesi.toml`, `[modelo.simulink]` / `[modelo.completo]` |
| perfil, solver, paso, duración, condiciones iniciales, actuadores | el script (`02_simulacion.py`, secciones 1 a 4) |
| controladores: ganancias y funciones `pitch_control`, `roll_control`, `yaw_control` | `gliderjax/autopilot.py` (guía en [CONTROL_Y_MISION.md](CONTROL_Y_MISION.md), sección 5) |
| constantes de la FSM de waypoints | `gliderjax/navigation.py`, líneas 20-36 |
| ecuaciones del modelo | `gliderjax/dynamics.py` (mapa en [MODELO.md](MODELO.md)) |

El archivo de configuración se revisa al cargar: si falta o sobra una
clave (por ejemplo un error de dedo), da error con el nombre.

## Cómo correr

Desde `GLIDER_JAX\sim\gliderjax`:

```
..\Scripts\python.exe escenarios\00_balance_estatico.py
..\Scripts\python.exe escenarios\01_validacion_dinamica.py
..\Scripts\python.exe escenarios\02_simulacion.py
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
