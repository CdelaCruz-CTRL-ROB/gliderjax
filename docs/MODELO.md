# Modelo físico: inventario y origen de cada parte

Última actualización: 2026-10-24

## 1. Inventario: qué viene de Simulink y qué se agregó

"Verificado" = comparado línea por línea contra tu código de Simulink.
"Según el port" = lo dicen los comentarios que dejó la sesión anterior, aún
no comparado contra tu código.

| Componente | Origen | Dónde | Cómo se activa |
|---|---|---|---|
| Modelo hidrodinámico en marco del flujo (D, SF, L, T1, T2, T3) | Simulink, **verificado** (sección 2) | `dynamics.py::tau_flowframe` | siempre |
| KMR · sign(α) (momento de roll) | Simulink, **verificado** | `tau_flowframe` | `kmr_mode="signum"` (default) |
| Amortiguamiento rotacional D_body | Simulink, **verificado** | `dynamics.py::tau_rot_damping` | `use_rot_damping=True` (default) |
| Presupuesto de 4 masas, M_RB, C_RB | Simulink (Mbr+Ma fcn), según el port | `dynamics.py::mass_geometry, M_rigid, C_rigid` | siempre |
| Masa añadida MA (diagonal) | Simulink, según el port (fuera de diagonal en cero) | `params.py` (`MA`) | siempre |
| VBD (pistón, flotabilidad) | Simulink (fcn_VBD_actuator), según el port | `dynamics.py::vbd_outputs, dxv_from_dB_ref` | siempre |
| Cadenas de actuadores (TFs, limitadores de tasa) | Simulink, según el port; **varios valores marcados `CONFIRM`** | `params.py::cidesi_defaults`, `sim.py::move_actuators` | siempre |
| Fuerzas de restauración (peso + flotación) | Reporte, g(η), según el port | `dynamics.py::tau_restoring` | siempre |
| Cableado de la perturbación dist_xg/dist_yg | Réplica del cableado de Simulink, según el port | `dynamics.py::dynamics` | `replicate_dist_wiring=True` |
| **Saturación tanh de α y β** (`A_MAX = 0.44`) | **Agregado por el port. NO está en tu Simulink** | `tau_flowframe`, líneas 204-206 | **siempre, sin switch** |
| **Crossflow** (arrastre por flujo cruzado) | **Agregado por el port** | `dynamics.py::tau_crossflow` | `use_crossflow=True` (default) |
| **Modelo de superficie** | **Agregado por el port** | `dynamics.py::surface_state, tau_surface` | `use_surface_model=True` (default) |
| **Geodesia** (posición ECEF verdadera) | **Agregado por el port** | `dynamics.py`, `geodesy.py` | `use_geodesy=True` (default) |
| Bloques C_A (Coriolis de masa añadida) | Agregado (estudios académicos) | `dynamics.py::C_added` | `use_ca_omega`, `use_ca_translational` (default False) |
| Amortiguamiento ∝ V (KpV, KqV, KrV) | Agregado, valores en 0 | `tau_rot_damping` | siempre, sin efecto |

### Importante para validar contra Simulink

Con `ModelConfig()` por defecto, superficie, crossflow y geodesia están
**encendidos**, así que **no** es el modelo de Simulink. Para reproducir
Simulink:

```python
cfg = ModelConfig(use_surface_model=False, use_crossflow=False, use_geodesy=False)
```

Con esos tres en `False`, la versión de trabajo da resultados **idénticos bit
a bit** a la v1 sin waypoints (verificado 2026-10-24, ver bitácora).

Aun así queda una diferencia con Simulink: la saturación tanh (sección 3).

## 2. Comparación con tu función de Simulink

Referencia: [referencia_simulink/fcn_damping.m](referencia_simulink/fcn_damping.m)

| Parte | Simulink | Port (`dynamics.py`) | ¿Igual? |
|---|---|---|---|
| D_body roll | `0.2*20*p + 0.2*120*p\|p\|` | `Kp_lin=4, Kp_quad=24` | sí |
| D_body pitch | `0.5*20*q + 0.5*100*q\|q\|` | `Kq_lin=10, Kq_quad=50` | sí |
| D_body yaw | `100*r\|r\|` | `Kr_lin=0, Kr_quad=100` | sí |
| Coeficientes | KD0 5.9, KD 214.62, Kbeta −56.66, KL0 0.326713·0, Kalpha 308.9754, KMR 4.5194, KM0 −0.0328, KM −12.48, KMYLIN 15.04, KMYCUB −398.442 | mismos valores (`KL` = Kalpha, `KL0` = 0) | sí |
| α, β | `atan2(w,u)`, `asin(v/V)` | igual (protección numérica si u≈0 o V≈0) | sí |
| KMR | `KMR*sign(alpha)` | `kmr_mode="signum"`, usa α sin saturar | sí |
| D, SF, L, TDL1, TDL2, TDL3 | usan **α y β directos** | usan **α_e, β_e saturados** | **NO** |
| R_CB (flujo → cuerpo) | matriz de tu código | `R_flow_to_body`, misma matriz | sí |
| Fuerzas y momentos rotados con R_CB | sí | sí | sí |
| Signo | `D_flow = −[F;T]`, se resta en la ecuación | suma `+[F;T]` al lado derecho | equivalente, *asumiendo* que tu diagrama resta `D_total` |

**Conclusión:** el port de tu función es exacto salvo la saturación.

Partes de tu función que no se usan (ni en Simulink ni en el port):
`R_EB`, la entrada `r_cw` y los modelos `D_B2` comentados.

## 3. Saturación tanh de α y β

```python
A_MAX = 0.44                              # rad, ~25°
alpha_e = A_MAX * tanh(alpha / A_MAX)     # ≈ alpha si alpha es chico
beta_e  = A_MAX * tanh(beta  / A_MAX)
```

**Qué hace:** limita suavemente α y β a ~25° dentro de las fórmulas de
D, SF, L y los momentos. Se aplica siempre (sumergido o no), pero solo cambia
algo cuando α o β son grandes.

**Por qué se puso:** no hay registro de la sesión anterior. Por el README y
los comentarios, fue una protección para que los polinomios ajustados del
CFD (α², β³) no extrapolaran fuera de su rango, pensada como temporal
hasta tener el crossflow. Quedaron las dos cosas.

**Efecto medido (2026-10-24), con y sin saturación:**

| Escenario | α máximo | Diferencia |
|---|---|---|
| Planeo sumergido, 400 s desde 50 m | 3° (estable 1.3°) | 0.01 m de profundidad |
| Arranque en superficie desde reposo, modelo Simulink | 83° | 0.18 m |
| Igual, con superficie + crossflow | 41° | 0.15 m |
| Misión triángulo con FSM, 6000 s | ~178° (con V≈0) | 1.8 m prof., 0.7 m horiz. |

- El planeador planea a α ≈ 1.3°, donde la saturación es despreciable.
- En la misión, α solo pasó de 6° a menos de 3.3 m de profundidad
  (arranque, fase SUB casi detenido, llegada a superficie). En las
  inversiones del fondo (~135 m) no pasó de 6°.
- Sin saturación no hubo NaN ni inestabilidad en ningún caso. Pero β nunca
  pasó de 3°, así que el término cúbico de yaw (−398·β³) no se probó con
  giros cerrados.

**Pendiente:** convertirla en switch, apagado por defecto (= Simulink).

**Contexto:** fuera de α pequeño, el modelo en términos de α y β no
representa bien la física (tema abierto del autor). La saturación, el
crossflow y el modelo de superficie son parches provisionales para ese
problema; ninguno sale de datos experimentales o CFD propios.

## 4. Crossflow (arrastre por flujo cruzado)

**No es la saturación ni tiene que ver con el giro.** Es una fuerza
adicional, fuera de tu función de damping.

**El nombre:** en hidrodinámica de cuerpos esbeltos (submarinos, misiles),
"crossflow" es la componente de la velocidad del agua **perpendicular al eje
del casco** (v y w en ejes cuerpo). A gran ángulo de ataque, esa componente
hace que el casco se comporte como un cilindro en flujo transversal, con un
arrastre ∝ ½·ρ·Cd·A·(V_perp)² (modelo clásico de Allen–Perkins).

**Fórmula** (`dynamics.py::tau_crossflow`):

```
Y = −Cd_cross_y · v·|v|        Z = −Cd_cross_z · w·|w|        (sin momentos)
peso = ½·(1 + tanh((χ − chi_blend) / chi_width)),   χ = acos(|u|/V)
```

| Incidencia total χ | Peso |
|---|---|
| 5° (planeo normal) | 0.5 % |
| 10° | 2 % |
| 23° | 50 % |
| ≥ 35° | ~100 % |

- Se **suma** al modelo del flujo; no lo reemplaza.
- `Cd_cross_y = Cd_cross_z = 200 kg/m`, `chi_blend = 0.40 rad`,
  `chi_width = 0.12 rad`: **estimados por geometría** (½·ρ·Cd·A con casco
  ~1.8 × 0.2 m), no medidos ni de CFD.
- Solo fuerzas, aplicadas en el origen: no produce momentos.

## 5. Modelo de superficie

Solo en la versión con waypoints. `dynamics.py::surface_state, tau_surface`.

- Calcula qué fracción del casco está fuera del agua (9 puntos a lo largo).
- Escala la flotabilidad por la fracción sumergida y mueve el centro de
  flotación al centroide sumergido.
- Reduce la masa añadida al emerger (mínimo `ma_scale_floor = 0.3`).
- Agrega cerca de la superficie: fuerza de entrada al agua y amortiguamiento
  de radiación en heave y pitch.

**Efecto principal:** con flotabilidad positiva el planeador sube hasta
quedar parcialmente fuera y se queda flotando (en la misión, ~0.1 m en
FIX). Sin este modelo (v1 NO_WP) subiría indefinidamente hacia z negativo.

Parámetros **estimados**, no medidos: `hull_L = 1.8`, `hull_H = 0.2`,
`B33_rad = 120`, `B55_rad = 40`, `dmA_dz = 120`. Se pueden identificar con
una prueba de oscilación libre en alberca.

## 6. Inversión del giro entre bajada y subida

**Comportamiento esperado (autor):** con roll a la izquierda, bajando gira a
la izquierda y subiendo a la derecha (relación roll–guiñada positiva al
bajar, negativa al subir).

**Cómo está representado:**

1. **Planta:** `KMR · sign(α)` en el momento de roll (`kmr_mode="signum"`,
   default). Como α > 0 al bajar y α < 0 al subir, el signo cambia.
2. **Control:** la FSM entrega una ganancia de acoplamiento `cg`
   (+1 en DIVE, −1 en CLIMB, 0 en los demás estados) que multiplica el roll
   deseado en el lazo de yaw. Ver [CONTROL_Y_MISION.md](CONTROL_Y_MISION.md).

**Prueba (2026-10-24):** masa rotatoria fija en γ = +30° (roll a la
izquierda), bajada 400 s y luego subida 400 s, modelo Simulink. Convención
NED: yaw rate negativo = gira a la izquierda.

| kmr_mode | Fase | Roll | α | Yaw rate | Giro neto en 240 s |
|---|---|---|---|---|---|
| signum (Simulink) | bajada | −25.9° | +1.25° | −0.83 °/s (izq.) | −287° |
| signum (Simulink) | subida | −25.7° | −1.34° | +0.91 °/s (der.) | +309° |
| const (sin sign α) | bajada | −25.9° | +1.25° | −0.83 °/s (izq.) | −287° |
| const (sin sign α) | subida | −24.6° | −1.31° | +0.63 °/s (der.) | +211° |

**Conclusiones:**

- **Sí está reflejado:** bajando gira a la izquierda, subiendo a la derecha.
- **La inversión ocurre aun sin sign(α).** En este modelo la dirección del
  giro cambia por sí sola entre bajada y subida (la sustentación, inclinada
  por el roll, apunta a lados opuestos). El sign(α) en KMR cambia **cuánto**
  gira al subir (~45 % más rápido: 0.91 vs 0.63 °/s), no la dirección.
- Vale la pena revisar si el sign(α) era necesario en Simulink o si
  compensaba otra cosa.

## 7. Switches del modelo (`ModelConfig` en `params.py`)

Cambiar cualquiera obliga a recompilar (~5–10 s).

| Switch | Default | Modelo Simulink | Qué hace |
|---|---|---|---|
| `kmr_mode` | `"signum"` | `"signum"` | `"signum"` = KMR·sign(α); `"parity"` = variante suave; `"const"` = sin dependencia de α |
| `kmy_cubic` | True | True | término cúbico en el momento de yaw |
| `use_rot_damping` | True | True | D_body |
| `use_surface_model` | True | **False** | sección 5 |
| `use_crossflow` | True | **False** | sección 4 |
| `use_geodesy` | True | **False** | posición ECEF (necesaria para la misión con GPS) |
| `use_ca_omega` | False | False | Coriolis de masa añadida (rotacional) |
| `use_ca_translational` | False | False | bloque de Munk: **duplica** física ya incluida en el CFD |
| `replicate_dist_wiring` | True | True | la perturbación de CG entra en restauración y C_RB pero no en M_RB, como en Simulink |
