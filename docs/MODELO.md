# Modelo físico: inventario y origen de cada parte

Última actualización: 2026-10-24

## 1. Inventario: qué viene de Simulink y qué se agregó

"Verificado" = comparado línea por línea contra tu código de Simulink.
"Según el port" = lo dicen los comentarios que dejó la sesión anterior, aún
no comparado contra tu código.

| Componente | Origen | Dónde | Cómo se activa |
|---|---|---|---|
| Modelo hidrodinámico en marco del flujo (D, SF, L, T1, T2, T3) | Simulink, **verificado** (sección 2.1) | `dynamics.py::tau_flowframe` | siempre |
| KMR · sign(α) (momento de roll) | Simulink, **verificado** | `tau_flowframe` | `kmr_mode="signum"` (default) |
| Amortiguamiento rotacional D_body | Simulink, **verificado** | `dynamics.py::tau_rot_damping` | `use_rot_damping=True` (default) |
| Presupuesto de 4 masas, M_RB, I_b, r_g | Simulink (Mbr+Ma fcn), **verificado** (sección 2.2), salvo la rotación de la inercia de la masa rotatoria | `dynamics.py::mass_geometry, M_rigid` | `rot_inertia_mode` (default `"RIRt"`; Simulink = `"RtIR"`) |
| C_RB (Coriolis de cuerpo rígido) | Simulink (calculate_coriolis_total_fossen), **verificado** (sección 2.3) | `dynamics.py::C_rigid` | siempre |
| Masa añadida MA | Simulink, **verificado**: tu código tiene términos fuera de la diagonal (0.1 kg, 0.01 kg·m²); la librería usa solo la diagonal por defecto | `params.py` (`MA` diagonal, `MA_SIMULINK` completa) | `p._replace(MA=MA_SIMULINK)` |
| VBD (pistón, flotabilidad) | Simulink (fcn_VBD_actuator), según el port | `dynamics.py::vbd_outputs, dxv_from_dB_ref` | siempre |
| Cadenas de actuadores (TFs, limitadores de tasa) | Simulink, según el port; **varios valores marcados `CONFIRM`** | `params.py::cidesi_defaults`, `sim.py::move_actuators` | siempre |
| Fuerzas de restauración (peso + flotación) | Reporte, g(η), según el port | `dynamics.py::tau_restoring` | siempre |
| Cableado de la perturbación dist_xg/dist_yg | Simulink, **verificado** en Mbr+Ma fcn: entra en r_g (C_RB, restauración), no en M ni en I_b | `dynamics.py::dynamics` | `replicate_dist_wiring=True` |
| **Saturación tanh de α y β** (`A_MAX = 0.44`) | **Agregado por el port. NO está en tu Simulink** | `tau_flowframe` | `use_alpha_saturation` (default **False** desde 2026-10-24) |
| **Crossflow** (arrastre por flujo cruzado) | **Agregado por el port** | `dynamics.py::tau_crossflow` | `use_crossflow=True` (default) |
| **Modelo de superficie** | **Agregado por el port** | `dynamics.py::surface_state, tau_surface` | `use_surface_model=True` (default) |
| **Geodesia** (posición ECEF verdadera) | **Agregado por el port** | `dynamics.py`, `geodesy.py` | `use_geodesy=True` (default) |
| Bloques C_A (Coriolis de masa añadida) | Tu Simulink lo calcula pero lo anula (`C_RB + C_A*0`), **verificado**. El port lo puede encender, pero con otra matriz (`MA_hull`) | `dynamics.py::C_added` | `use_ca_omega`, `use_ca_translational` (default False = Simulink) |
| Amortiguamiento ∝ V (KpV, KqV, KrV) | Agregado, valores en 0 | `tau_rot_damping` | siempre, sin efecto |

### Importante para validar contra Simulink

Con `ModelConfig()` por defecto, superficie, crossflow y geodesia están
**encendidos**, así que **no** es el modelo de Simulink. Para reproducir
Simulink:

```python
from gliderjax.params import cidesi_defaults, MA_SIMULINK
p = cidesi_defaults()._replace(MA=MA_SIMULINK)
cfg = ModelConfig(use_surface_model=False, use_crossflow=False,
                  use_geodesy=False, rot_inertia_mode="RtIR")
```

(`use_alpha_saturation` ya es `False` por defecto.) El script
`escenarios/01_validacion_dinamica.py` escribe todos los switches
explícitamente; ver [VALIDACION.md](VALIDACION.md).

Verificado bit a bit (2026-10-24): con la saturación encendida y los otros
tres apagados, la versión de trabajo da lo mismo que la v1 sin waypoints.

### Diferencias conocidas con Simulink

1. **Inercia de la masa rotatoria** y **términos fuera de la diagonal de
   MA**: ver sección 2.2. Las dos tienen opción para replicar Simulink y su
   efecto es despreciable.
2. **Flotación en superficie** (según comentarios del port, **no
   verificada**). El comentario de `surface_state` dice que reemplaza una
   `fcn_buoyancy_surface` "de dos puntos" de tu Simulink. En la
   configuración equivalente a Simulink (superficie apagada) esa función no
   está representada, así que las pruebas deben quedarse lejos de z = 0.
3. **Entradas que vienen de otros bloques** (no verificadas): la posición
   de la masa móvil `r_lin = [0.2861 + rc, 0, 0.001]` y la del pistón
   `r_p = [−0.7 + Δx, 0, 0]` (de `fcn_VBD_actuator`).

Para cerrarlas: comparar contra `fcn_VBD_actuator`, `fcn_buoyancy_surface`
y el bloque que arma `r_lin`.

## 2. Comparación con tus funciones de Simulink

### 2.1 Damping

Referencia: [referencia_simulink/fcn_damping.m](referencia_simulink/fcn_damping.m)

| Parte | Simulink | Port (`dynamics.py`) | ¿Igual? |
|---|---|---|---|
| D_body roll | `0.2*20*p + 0.2*120*p\|p\|` | `Kp_lin=4, Kp_quad=24` | sí |
| D_body pitch | `0.5*20*q + 0.5*100*q\|q\|` | `Kq_lin=10, Kq_quad=50` | sí |
| D_body yaw | `100*r\|r\|` | `Kr_lin=0, Kr_quad=100` | sí |
| Coeficientes | KD0 5.9, KD 214.62, Kbeta −56.66, KL0 0.326713·0, Kalpha 308.9754, KMR 4.5194, KM0 −0.0328, KM −12.48, KMYLIN 15.04, KMYCUB −398.442 | mismos valores (`KL` = Kalpha, `KL0` = 0) | sí |
| α, β | `atan2(w,u)`, `asin(v/V)` | igual (protección numérica si u≈0 o V≈0) | sí |
| KMR | `KMR*sign(alpha)` | `kmr_mode="signum"`, usa α sin saturar | sí |
| D, SF, L, TDL1, TDL2, TDL3 | usan **α y β directos** | α y β directos por defecto; saturados solo con `use_alpha_saturation=True` | sí (desde 2026-10-24; antes **no**) |
| R_CB (flujo → cuerpo) | matriz de tu código | `R_flow_to_body`, misma matriz | sí |
| Fuerzas y momentos rotados con R_CB | sí | sí | sí |
| Signo | `D_flow = −[F;T]`, se resta en la ecuación | suma `+[F;T]` al lado derecho | equivalente, *asumiendo* que tu diagrama resta `D_total` |

**Conclusión:** el port de tu función es exacto salvo la saturación, que
desde 2026-10-24 está apagada por defecto. Con `use_alpha_saturation=False`
el port coincide con tu función.

Partes de tu función que no se usan (ni en Simulink ni en el port):
`R_EB`, la entrada `r_cw` y los modelos `D_B2` comentados.

### 2.2 Masas: `Mbr+Ma fcn`

Referencia: [referencia_simulink/fcn_Mbr_Ma.m](referencia_simulink/fcn_Mbr_Ma.m)
(revisado 2026-10-24)

| Parte | Simulink | Port | ¿Igual? |
|---|---|---|---|
| Masas | m_s 37, m_lin 15, m_rot 5, m_p 1 | `m_static`, `m_lin`, `m_rot`, `m_pist` | sí |
| r_s | [−0.1241, 0, 0.001] | `r_static` | sí |
| r_rot | [0.2, R·cos(γ+π/2), R·sin(γ+π/2)], R = 0.07 | [0.2, −R·sin γ, R·cos γ] | sí (misma expresión) |
| I_s, I_lin, I_rot, I_p | diagonales de tu código | mismos valores | sí |
| Inercia de la masa rotatoria en cuerpo | `Rot_xᵀ·I_rot·Rot_x` | `Rx·I_rot·Rxᵀ` (default) | **no**, ver abajo |
| M_RB | `[m·I, −Σ mᵢ s(rᵢ); Σ mᵢ s(rᵢ), I_b]` | `M_rigid(m, r_g, I_b)`, con `m·s(r_g) = Σ mᵢ s(rᵢ)` | sí |
| I_b | `Σ (Iᵢ − mᵢ s(rᵢ)²)` | `Σ about_CO(Iᵢ, mᵢ, rᵢ)` | sí |
| r_g | `Σ mᵢ rᵢ / m` + dist_xg, dist_yg | `r_g_dist` | sí |
| dist en M e I_b | no entra (M se arma con posiciones sin perturbar) | `replicate_dist_wiring=True` | sí |
| MA, IA | diagonal + 0.1 (MA) y 0.01 (IA) fuera de la diagonal | solo diagonal | **no**, ver abajo |
| CA (acoplamiento traslación–rotación de M_A) | 0 (`2.57*0`) | 0 | sí |
| M = M_RB + M_A | sí | sí | sí |

**Diferencia 1: rotación del tensor de la masa rotatoria.**

- Tu posición `r_rot = Rot_x·[0.2, 0, R]`, así que la masa gira con
  `Rot_x`. Para ser consistente, su tensor debería girar igual:
  `Rot_x·I·Rot_xᵀ`. Tu código usa `Rot_xᵀ·I·Rot_x`, que equivale a girar
  el tensor −γ.
- Como I_rot es diagonal, las dos formas dan **la misma diagonal**. Solo
  cambia el **signo** del término cruzado yz de la masa rotatoria,
  ±(I_zz − I_yy)·sin γ·cos γ.
- En el I_b total, el término yz vale 0.0112 (port) contra 0.0133
  (Simulink) a γ = 45°. La diferencia máxima es 0.0022 kg·m², frente a
  I_yy ≈ 30 kg·m².
- Switch: `rot_inertia_mode="RtIR"` replica tu código; el default
  `"RIRt"` es el consistente con `r_rot`.

**Diferencia 2: términos fuera de la diagonal de la masa añadida.**

- Tu código: 0.1 kg en MA (acoplan u–v, u–w, v–w) y 0.01 kg·m² en IA
  (acoplan p–q, p–r, q–r). Tienes comentada la versión diagonal.
- El port los puso en cero ("leftovers removed", según su comentario). Por
  simetría babor–estribor, los acoplamientos u–v, v–w, p–q y q–r deberían
  ser cero; u–w y p–r podrían no serlo.
- Opción: `p._replace(MA=MA_SIMULINK)` (constante en `params.py`).
- **Pregunta abierta:** ¿qué representan esos 0.1 y 0.01? ¿Valores
  medidos, de CFD o un relleno?

**Efecto medido** (escenario de validación, 1200 s, contra el port por
defecto):

| Variante | Posición horizontal | Profundidad | Roll | Pitch | Yaw |
|---|---|---|---|---|---|
| `RtIR`, MA diagonal | 0.0003 m | < 0.0001 m | 0.0006° | 0.0009° | 0.0009° |
| `RIRt`, `MA_SIMULINK` | 0.38 m | 0.0007 m | 0.008° | 0.005° | 0.13° |
| las dos (= Simulink) | 0.38 m | 0.0007 m | 0.008° | 0.006° | 0.13° |

Los valores cuasi-estacionarios por fase no cambian en ningún decimal
mostrado: con ν̇ = 0, la matriz de masa no influye. Las diferencias vienen
de los transitorios.

### 2.3 Coriolis: `calculate_coriolis_total_fossen`

Referencia:
[referencia_simulink/calculate_coriolis_total_fossen.m](referencia_simulink/calculate_coriolis_total_fossen.m)
(revisado 2026-10-24)

| Parte | Simulink | Port | ¿Igual? |
|---|---|---|---|
| C_RB | `[m S(ω), −m S(ω) S(r_g); m S(r_g) S(ω), −S(I_b ω)]` | `C_rigid`, misma matriz | sí |
| r_g usado en C_RB | el de Mbr+Ma fcn (con dist) | `r_g_dist` | sí |
| C_A | `[0, −S(a1); −S(a1), −S(a2)]`, a1 = MA·v₁ + CA·v₂, a2 = CAᵀ·v₁ + IA·v₂ | `C_added`, misma estructura | sí |
| C_A en la ecuación | **anulado**: `C_total = C_RB + C_A*0` | apagado por defecto (`use_ca_*=False`) | sí |
| Matriz usada en C_A si se enciende | MA, IA de Mbr+Ma fcn | `MA_hull` (otra matriz, no viene de Simulink) | no, pero apagado |

**Conclusión:** C_RB es idéntico y C_A está apagado en ambos. Sin C_A no
hay momento de Munk explícito. El comentario del port dice que el CFD ya
lo incluye en KM, así que encenderlo lo contaría dos veces.

No verificado: qué velocidad entra a esta función en tu diagrama (ν o
ν_r). Sin corriente son iguales, y la validación corre sin corriente.

## 3. Saturación tanh de α y β

```python
A_MAX = 0.44                              # rad, ~25°
alpha_e = A_MAX * tanh(alpha / A_MAX)     # ≈ alpha si alpha es chico
beta_e  = A_MAX * tanh(beta  / A_MAX)
```

**Estado:** switch `use_alpha_saturation` en `ModelConfig`, **apagado por
defecto** (= Simulink) desde 2026-10-24.

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
| `use_alpha_saturation` | False | False | sección 3 |
| `rot_inertia_mode` | `"RIRt"` | **`"RtIR"`** | rotación del tensor de la masa rotatoria, sección 2.2 |
| `use_rot_damping` | True | True | D_body |
| `use_surface_model` | True | **False** | sección 5 |
| `use_crossflow` | True | **False** | sección 4 |
| `use_geodesy` | True | **False** | posición ECEF (necesaria para la misión con GPS) |
| `use_ca_omega` | False | False | Coriolis de masa añadida (rotacional) |
| `use_ca_translational` | False | False | bloque de Munk: **duplica** física ya incluida en el CFD |
| `replicate_dist_wiring` | True | True | la perturbación de CG entra en restauración y C_RB pero no en M_RB, como en Simulink |
