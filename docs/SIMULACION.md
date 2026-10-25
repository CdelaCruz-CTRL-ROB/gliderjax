# Balance estático y simulación manual

Última actualización: 2026-10-24

Dos scripts para probar a mano:

| Script | Para qué |
|---|---|
| [`escenarios/00_balance_estatico.py`](../escenarios/00_balance_estatico.py) | sin simular: masas, flotabilidad, CG, **cero virtual**, actitud en reposo |
| [`escenarios/02_simulacion.py`](../escenarios/02_simulacion.py) | simulación por tiempo: eliges solver, paso, condiciones iniciales y actuadores; ves gráficas; se compara con la corrida anterior |

Los dos leen los parámetros de [`config/cidesi.toml`](../config/cidesi.toml).
Desde `GLIDER_JAX\sim\gliderjax`:

```
..\Scripts\python.exe escenarios\00_balance_estatico.py
..\Scripts\python.exe escenarios\02_simulacion.py
```

Flujo sugerido: cambiar masas o volumen en el `.toml` → `00` para ver si
quedó neutro y dónde está el cero virtual → `02` para ver cómo vuela.

## 1. Balance estático (`00_balance_estatico.py`)

Entradas (sección 2 del script): `RC` [m], `GAMMA_DEG`, `DB` [N], y los
rangos de las curvas.

Imprime:

| Bloque | Contenido |
|---|---|
| MASAS | cada masa con su posición para esos actuadores; masa total y peso |
| FLOTABILIDAD | B con dB = 0 (ρ·g·V_displaced) y con el dB pedido; flotabilidad neta; **V_displaced para ser neutro**; si el VBD alcanza a compensar |
| CENTROS | CG (con perturbaciones), CB, BG y si es estable |
| CERO VIRTUAL NOMINAL | sin perturbaciones: posición de la masa móvil para pitch 0 en reposo (dB = 0, γ = 0), comparada con `r_lin0` del archivo, y el rc que hace falta; γ para roll 0 |
| EFECTO DE LA MASA DE PERTURBACIÓN | solo si `m_dist` ≠ 0: error de cero equivalente (mm de masa móvil, grados de γ); rc para pitch 0, pitch en reposo y γ para roll 0, nominal contra real |
| EQUILIBRIO | roll y pitch en reposo con los actuadores pedidos |

Figura `out/00_balance_estatico/curvas_reposo.png`: pitch en reposo vs rc
(una curva por dB) y roll en reposo vs γ.

"En reposo" = solo peso y flotación, sin velocidad. Usa las mismas
funciones que la simulación (`mass_geometry`, `vbd_outputs`,
`tau_restoring`). En planeo el pitch es otro por el momento hidrodinámico.

**Valores con el archivo actual (2026-10-24):**

- Masa 58 kg, V_displaced 0.058 m³: neutro (B − W = 0).
- CG: x_g = −0.003 mm, z_g = +6.93 mm (BG = 6.9 mm, estable pero blando).
- Cero virtual calculado: **0.28611 m**; en el archivo `r_lin0 = 0.2861`:
  con rc = 0 el pitch en reposo es +0.03°.
- Sensibilidad cerca de rc = 0: ~1° de pitch en reposo por mm de masa
  móvil. Un error de 1 mm en el cero virtual se nota.
- Con dB = −3 N (bajar) el pitch en reposo con rc = 0 es ~+20° (nariz
  arriba): el pistón quita flotación en la cola.
- γ = 30°: roll en reposo −26° (igual al cálculo analítico atan(y_g/z_g)).

## 2. Simulación manual (`02_simulacion.py`)

### Entradas

| Sección | Variable | Qué es |
|---|---|---|
| 1 Modelo | `CONFIG`, `MODELO` | archivo de parámetros y perfil de switches (`simulink` o `completo`) |
| 2 Solver | `SOLVER` | `"ode4"` (Runge-Kutta 4), `"ode2"` (Heun), `"ode1"` (Euler); paso fijo, nombres de Simulink |
| | `DT` | paso [s]; default 0.005 |
| | `T_FINAL` | duración [s] |
| | `MUESTREO` | cada cuánto se guarda una muestra [s], múltiplo de `DT` |
| 3 Condiciones iniciales | `POSICION` | x, y, z en NED [m] |
| | `ACTITUD_DEG` | roll, pitch, yaw [°] |
| | `VELOCIDAD_MARCO`, `VELOCIDAD` | `"cuerpo"`: (u, v, w) relativa al agua; `"inercial"`: (vN, vE, vD) respecto al fondo (se le resta la corriente) |
| | `VEL_ANGULAR_DEG` | p, q, r [°/s] |
| | `ORIGEN_GEO` | lon, lat [°]; solo si el perfil usa geodesia |
| 4 Actuadores | `MODO` | `"directo"`: la tabla da la posición de los actuadores; `"pitch_roll"`: referencias de pitch y roll, cada controlador por su lado; `"rumbo"`: referencias de pitch y rumbo, el control de yaw da la referencia de roll |
| | `FASES` | modo directo: (t_inicio, rc [m], γ [°], dB [N], nombre); una fila = constante |
| | `FASES_PITCH_ROLL` | modo pitch_roll: (t_inicio, pitch_ref [°], roll_ref [°], dB [N], nombre) |
| | `FASES_RUMBO` | modo rumbo: (t_inicio, pitch_ref [°], rumbo_ref [°], dB [N], signo_giro, nombre). signo_giro: +1 bajando, −1 subiendo, 0 sin control de rumbo |
| | `PERIODO_CONTROL` | cada cuánto corren los controladores [s], múltiplo de `DT` |
| | `ACTUADORES_INICIO` | `"en_posicion"`: arrancan ya en la primera fila (modos con control: rc = 0, γ = 0 y el dB de la primera fila); `"neutro"`: arrancan en 0 y se mueven con su dinámica |
| 5 Salida | `FIGURAS` | lista de figuras (comentar las que no quieras) |
| | `COMPARAR_CON_ANTERIOR` | superponer con la corrida anterior |
| | `ALFA_VALIDO_DEG` | aviso si \|α\| pasa de esto |
| | `MOSTRAR` | abrir las figuras al final |

### Qué hace al correr

1. Imprime modelo, solver, número de pasos y la flotabilidad neta con
   dB = 0 (para ver de inmediato si las masas y el volumen no cuadran). Si
   hay masa de perturbación, imprime su error de cero equivalente.
2. Simula. Con dt = 0.005 s, 600 s tardan ~4 s.
3. Guarda en `out/02_simulacion/`: `telemetria.npz`, `entradas.json`
   (todas las variables de las secciones 1 a 4) y `config_usada.toml`. La
   corrida anterior pasa a `out/02_simulacion/anterior/`.
4. **Imprime qué cambió** respecto a la corrida anterior: entradas del
   script, parámetros del `.toml` y switches. Ejemplo:
   ```
   cambios respecto a la corrida anterior:
     MODELO: simulink -> completo
     VELOCIDAD: [0.0, 0.0, 0.0] -> [0.3, 0.0, 0.2]
     use_crossflow: False -> True  (switch)
   ```
5. Avisos: NaN, \|α\| fuera de rango, superficie sin modelo, rc o dB fuera
   de la saturación.
6. Resumen por fase y estado final.
7. Figuras, más `00_vs_anterior.png` con las dos corridas superpuestas.

### Modos con control

Usan los controladores **provisionales** de `autopilot.py` (no tus bloques
de Simulink): PID de pitch → rc, P de roll → γ, y en modo `"rumbo"` P de
rumbo → roll de referencia (× signo de giro). Las referencias pasan por el
moldeado de Simulink (pitch ±0.1 rad/s y 1/(5s+1); rumbo ±0.03 rad/s y
1/(5s+1); roll sin moldeado), que arranca en la actitud inicial. El dB va
directo al VBD. Cómo poner tus controladores:
[CONTROL_Y_MISION.md](CONTROL_Y_MISION.md), sección 5.

Agrega la figura `09_control.png`: pitch, roll y rumbo contra la
referencia **usada** por el controlador (después del moldeado; en modo
rumbo, la de roll es la que calculó el control de yaw) y contra la de la
tabla; rc y γ (comando y posición); y `mem[0]` del lazo de pitch (en el
provisional, el aporte integral, con líneas en ±`I_MAX_P`).

Pruebas (2026-10-24, controlador a 0.25 s, pitch con θ̇ − θ̇_ref y aporte
integral con tope; bitácora entradas 11 y 12), arrancando del reposo a
200 m con actuadores en neutro:

- **`pitch_roll`**, pitch_ref −25°, roll_ref 0° → 15° en t = 200 s:
  - pitch sobrepasa a −26.1° y se asienta en −25.000° (±0.5° desde
    t = 33 s);
  - roll se queda en **9.5°** con referencia 15°: error en estado
    estacionario del P provisional (sin integral);
  - con roll sostenido el planeador gira (+0.69 °/s).
- **`rumbo`**, rumbo_ref 0° → 45° en t = 200 s:
  - el rumbo pasa de 44° en t = 264 s y se queda en 45.0°; roll máximo
    16.8°;
  - al meter el roll, primero gira 1.3° al lado contrario;
  - pitch se mantiene entre −25.2° y −24.8° durante el giro.
- Con rumbo inicial 90° y referencia 90°, el rumbo se queda en 90.000°
  (sin giro falso).

### Paso de integración

Prueba de validación (1200 s, cuatro fases), diferencia máxima en pitch:

| Comparación | Máxima | En estacionario |
|---|---|---|
| dt 0.05 vs 0.005 | 0.54° (en el cambio de bajada a subida) | 0 |
| dt 0.005 vs 0.0025 | 0.03° | 0 |

dt = 0.005 está convergido para los transitorios. Para comparar con
Simulink, usar el **mismo solver y el mismo paso** en los dos.

Los tres solvers con dt = 0.005 difieren entre sí menos de 0.002° en
pitch.

### Cosas a tener en cuenta

- **Arrancar desde el reposo** (V = 0) con dB ≠ 0: al principio el
  planeador cae casi vertical, α ≈ 90° durante unos segundos y sale el
  aviso de α. Es real: el modelo α–β no vale ahí, aunque las fuerzas son
  chicas porque V es chica.
- **`ACTUADORES_INICIO = "neutro"`** incluye el transitorio de los
  actuadores (la masa móvil a 8 mm/s tarda ~4 s en llegar a 30 mm; el VBD
  a 0.5 N/s tarda 6 s en llegar a 3 N).
- **Corriente y perturbaciones** están en el `.toml`
  (`[perturbaciones]`), no en el script.
- Con masas distintas, los mismos comandos dan otro vuelo: revisar primero
  con `00_balance_estatico.py`.
- Para guardar una corrida que no quieres perder, copia la carpeta
  `out/02_simulacion/`: la siguiente corrida mueve la actual a
  `anterior/` y borra la que estaba ahí.
