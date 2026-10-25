# Modelo físico: inventario y origen de cada parte

Última actualización: 2026-10-24

## 1. Inventario: qué viene de Simulink y qué se agregó

"Verificado" = comparado línea por línea contra tu código de Simulink.
"Según el port" = lo dicen los comentarios que dejó la sesión anterior, aún
no comparado contra tu código.

**Todos los valores numéricos** están en
[`config/cidesi.toml`](../config/cidesi.toml), con unidades y origen de
cada uno. Los switches están en el mismo archivo, como perfiles
(`[modelo.simulink]`, `[modelo.completo]`).

| Componente | Origen | Dónde | Cómo se activa |
|---|---|---|---|
| Modelo hidrodinámico en marco del flujo (D, SF, L, T1, T2, T3) | Simulink, **verificado** (sección 2.1) | `dynamics.py::tau_flowframe` | siempre |
| KMR · sign(α) (momento de roll) | Simulink, **verificado** | `tau_flowframe` | `kmr_mode="signum"` (default) |
| Amortiguamiento rotacional D_body | Simulink, **verificado** | `dynamics.py::tau_rot_damping` | `use_rot_damping=True` (default) |
| Presupuesto de 4 masas, M_RB, I_b, r_g | Simulink (Mbr+Ma fcn, fcn_r_lin), **verificado** (secciones 2.2 y 2.5); inercia de la masa rotatoria **corregida** | `dynamics.py::mass_geometry, M_rigid` | siempre |
| C_RB (Coriolis de cuerpo rígido) | Simulink (calculate_coriolis_total_fossen), **verificado** (sección 2.3) | `dynamics.py::C_rigid` | siempre |
| Masa añadida MA | Simulink, **verificado**; solo la diagonal (los acoplamientos de 0.1 y 0.01 en 0, sección 2.2) | `config/cidesi.toml`, `[masa_anadida]` | siempre |
| VBD (pistón, flotabilidad) | Simulink (fcn_VBD_actuator), **verificado** (sección 2.4) | `dynamics.py::vbd_outputs, dxv_from_dB_ref` | siempre |
| Cadenas de actuadores (TFs, limitadores de tasa) | Simulink, según el port; **varios valores marcados CONFIRMAR** | `config/cidesi.toml`, `[actuadores]`; `sim.py::move_actuators` | siempre |
| Fuerzas de restauración (peso + flotación) | Reporte, g(η), según el port | `dynamics.py::tau_restoring` | siempre |
| Perturbación de CG (disturb_xg, disturb_yg) | Simulink, **verificado** en Mbr+Ma fcn: entra en r_g (C_RB, restauración), no en M ni en I_b | `dynamics.py::dynamics` | `replicate_disturb_wiring=True` |
| Perturbación de la masa móvil (disturb_xlin): error en el cero virtual del actuador | Simulink (fcn_r_lin, `distxlin`), **verificado** | `dynamics.py::mass_geometry` | siempre (0 = sin perturbación) |
| Superficie de 2 puntos (B_scale, Δx_b) | Simulink (fcn_buoyancy_surface), **verificado** (sección 2.6) | `dynamics.py::surface_two_point` | `surface_model="simulink"` |
| **Saturación tanh de α y β** (`A_MAX = 0.44`) | **Agregado por el port. NO está en tu Simulink** | `tau_flowframe` | `use_alpha_saturation` (default **False** desde 2026-10-24) |
| **Crossflow** (arrastre por flujo cruzado) | **Agregado por el port** | `dynamics.py::tau_crossflow` | `use_crossflow=True` (default) |
| **Superficie completa** (9 estaciones, masa añadida, entrada y radiación) | **Agregado por el port** | `dynamics.py::surface_state, tau_surface` | `surface_model="full"` (default) |
| **Geodesia** (posición ECEF verdadera) | **Agregado por el port** | `dynamics.py`, `geodesy.py` | `use_geodesy=True` (default) |
| Bloques C_A (Coriolis de masa añadida) | Tu Simulink lo calcula pero lo anula (`C_RB + C_A*0`), **verificado**. El port lo puede encender, pero con otra matriz (`MA_hull`) | `dynamics.py::C_added` | `use_ca_omega`, `use_ca_translational` (default False = Simulink) |
| Amortiguamiento ∝ V (KpV, KqV, KrV) | Agregado, valores en 0 | `tau_rot_damping` | siempre, sin efecto |

### Importante para validar contra Simulink

Usar el perfil `simulink` del archivo de configuración:

```python
from gliderjax.params import load_config
p, cfg = load_config("config/cidesi.toml", model="simulink", dt=0.05)
```

Así lo hace `escenarios/01_validacion_dinamica.py`; ver
[VALIDACION.md](VALIDACION.md). `ModelConfig()` sin argumentos **no** es el
modelo de Simulink: trae superficie completa, crossflow y geodesia
encendidos (es el perfil `completo`).

Verificado bit a bit (2026-10-24): con la saturación encendida y los otros
tres apagados, la versión de trabajo da lo mismo que la v1 sin waypoints.

### Diferencias con Simulink que se dejan a propósito

1. **Inercia de la masa rotatoria**: el port usa `R·I·Rᵀ`, la consistente
   con la posición de la masa; tu Simulink tiene `Rᵀ·I·R` (error). Decisión
   del autor, 2026-10-24: no replicar el error. Sección 2.2.
2. **Masa añadida**: solo la diagonal; tus acoplamientos de 0.1 y 0.01
   van en 0. Sección 2.2.
3. **p_ref de `fcn_VBD_actuator`** (pitch automático según el signo de dB):
   no se porta. El pitch de cada inmersión lo da la misión. Sección 2.4.

Falta verificar: el bloque que arma el vector de restauración (el port
usa `r_b = [x_b + Δx_b, 0, 0]` y `B·B_scale`, confirmado por el autor) y
las cadenas de actuadores (valores CONFIRMAR).

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
| Inercia de la masa rotatoria en cuerpo | `Rot_xᵀ·I_rot·Rot_x` | `Rx·I_rot·Rxᵀ` | **no, a propósito**: ver abajo |
| M_RB | `[m·I, −Σ mᵢ s(rᵢ); Σ mᵢ s(rᵢ), I_b]` | `M_rigid(m, r_g, I_b)`, con `m·s(r_g) = Σ mᵢ s(rᵢ)` | sí |
| I_b | `Σ (Iᵢ − mᵢ s(rᵢ)²)` | `Σ about_CO(Iᵢ, mᵢ, rᵢ)` | sí |
| r_g | `Σ mᵢ rᵢ / m` + dist_xg, dist_yg | `r_g_dist` (con `disturb_xg`, `disturb_yg`) | sí |
| dist en M e I_b | no entra (M se arma con posiciones sin perturbar) | `replicate_disturb_wiring=True` | sí |
| MA, IA | diagonal + 0.1 (MA) y 0.01 (IA) en todos los acoplamientos | solo la diagonal | **no, a propósito**: ver abajo |
| CA (acoplamiento traslación–rotación de M_A) | 0 (`2.57*0`) | 0 | sí |
| M = M_RB + M_A | sí | sí | sí |

**Diferencia 1: rotación del tensor de la masa rotatoria.**

- Tu posición `r_rot = Rot_x·[0.2, 0, R]`, así que la masa gira con
  `Rot_x`. Para ser consistente, su tensor debe girar igual:
  `Rot_x·I·Rot_xᵀ`. Tu código usa `Rot_xᵀ·I·Rot_x`, que equivale a girar
  el tensor −γ.
- Como I_rot es diagonal, las dos formas dan **la misma diagonal**. Solo
  cambia el **signo** del término cruzado yz de la masa rotatoria,
  ±(I_zz − I_yy)·sin γ·cos γ.
- En el I_b total, el término yz vale 0.0112 (port) contra 0.0133
  (Simulink) a γ = 45°. La diferencia máxima es 0.0022 kg·m², frente a
  I_yy ≈ 30 kg·m².
- **Decisión (autor, 2026-10-24):** usar la correcta, `R·I·Rᵀ`, sin opción
  para replicar el error.

**Diferencia 2: acoplamientos de la masa añadida.**

Dos ideas distintas:

- **La matriz siempre es simétrica** (M_A = M_Aᵀ), sea cual sea la
  geometría. Sale de la energía cinética del fluido (flujo potencial). Tu
  matriz ya lo era. El cargador del archivo de configuración rechaza una MA
  no simétrica.
- **La simetría de la geometría pone ceros.** Si el casco es simétrico
  babor–estribor (plano xz), los movimientos "longitudinales" (u, w, q) no
  se acoplan con los "laterales" (v, p, r): los términos u–v, v–w, p–q y q–r
  son **exactamente cero**. Con pequeñas diferencias de fabricación serían
  pequeños pero no cero; normalmente se desprecian porque no se pueden
  estimar con precisión.
- u–w y p–r **sí** pueden existir: el casco no es simétrico arriba–abajo
  (aleta vertical, antena).

**Decisión (autor, 2026-10-24):** todos los acoplamientos en 0, solo la
diagonal. (Primero se probó conservar u–w y p–r; el autor indicó que esos
valores no importan.) Si en el futuro hay datos de CFD para u–w, u–q,
w–q, p–r, v–p o v–r, se escriben en `[masa_anadida]` del archivo de
configuración.

**Efecto medido** (escenario de validación, 1200 s, contra MA diagonal):

| Variante | Posición horizontal | Profundidad | Roll | Pitch | Yaw |
|---|---|---|---|---|---|
| inercia `Rᵀ·I·R` de Simulink | 0.0003 m | < 0.0001 m | 0.0006° | 0.0009° | 0.0009° |
| MA solo con u–w y p–r | 0.012 m | 0.001 m | 0.004° | 0.001° | 0.010° |
| MA de Simulink completa | 0.38 m | 0.0007 m | 0.008° | 0.005° | 0.13° |

Los valores cuasi-estacionarios por fase no cambian en ningún decimal
mostrado: con ν̇ = 0, la matriz de masa no influye. Las diferencias vienen
de los transitorios. Lo que más pesa son los acoplamientos u–v y v–w que la
simetría elimina.

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

### 2.4 VBD: `fcn_VBD_actuator`

Referencia: [referencia_simulink/fcn_VBD_actuator.m](referencia_simulink/fcn_VBD_actuator.m)
(revisado 2026-10-24)

| Parte | Simulink | Port (`vbd_outputs`, `dxv_from_dB_ref`) | ¿Igual? |
|---|---|---|---|
| ρ, g | 1000, 9.81 | `rho` (archivo de configuración), `G0` | sí |
| B_neutral | `W_total = 58*9.81` (constante) | `rho·g·V_displaced`, `V_displaced = 0.058 m³` | sí; independiente de las masas, ver abajo |
| Área | `pi*(0.08/2)^2` | `pi*(vbd_diam/2)^2` | sí (antes `3.14159265*0.04^2`, diferencia relativa 1e-9) |
| Δx pedido y saturación | `−(dB/(ρg))/área`, ±carrera/2 | igual | sí |
| r_p | `[x0 + Δx, 0, 0]`, x0 = −0.7 | igual | sí |
| B_total | `B_neutral + ΔV·ρ·g`, ΔV = −Δx·área | igual | sí |
| x_b | `ΔB·(x0 + Δx/2)/B_total` (rb_neutral = 0) | igual | sí |
| `percent_fill`, `delta_V_actual` | salidas de monitoreo | no se calculan | no afecta la dinámica |
| `p_ref` | ±`con` según el signo de dB | **no se porta** | a propósito, ver abajo |

**Flotabilidad neutra: separada de las masas.** `B_neutral = ρ·g·V_displaced`,
con ρ y V_displaced en `[flotabilidad]` del archivo de configuración.
**Decisión (autor, 2026-10-24),** después de probar las dos opciones:

- Si la flotabilidad siguiera a las masas, el planeador sería neutro
  siempre y se perdería un efecto real: cambiar una masa (o la densidad
  del agua) lo saca de neutro.
- Para que sea neutro se ajusta a mano `V_displaced`.
  `escenarios/00_balance_estatico.py` dice cuánto debe valer (ver
  [SIMULACION.md](SIMULACION.md)), y `02_simulacion.py` imprime la
  flotabilidad neta al arrancar.
- Con los valores actuales (ρ = 1000, V = 0.058 m³) es igual que tu
  Simulink: B − W = 0 exacto.
- **Ojo con ρ:** los coeficientes hidrodinámicos (`[hidrodinamica]`) ya
  traen la densidad del CFD y no se escalan al cambiar `rho`; solo cambian
  la flotabilidad y la conversión N ↔ carrera del pistón.

**p_ref (pitch automático según el signo de dB).** Tu bloque daba un pitch
de referencia ±`con` al bajar o subir, y el control de pitch tenía un switch
para ignorarlo. El port no lo tiene: en la misión, el pitch de referencia
lo da la FSM (`PITCH_DN`, `PITCH_UP`). **Decisión (autor, 2026-10-24):**
el actuador no decide la guiada; el pitch se da explícitamente para cada
inmersión. Pendiente para el script de waypoints: pitch por inmersión como
parámetro de la misión.

Supuesto del port, sin verificar: `fcn_VBD_actuator` recibe dB **después**
de la dinámica del actuador (prefiltro, limitador, `1/(2s+1)`), así que el
pistón sigue a la flotabilidad filtrada. Ver los valores CONFIRMAR de
`[actuadores]`.

### 2.5 Posición de la masa móvil: `fcn_r_lin`

Referencia: [referencia_simulink/fcn_r_lin.m](referencia_simulink/fcn_r_lin.m)
(revisado 2026-10-24)

| Simulink | Port (`mass_geometry`) | ¿Igual? |
|---|---|---|
| `r_lin = [0.2861 + r_c + distxlin, 0, 0.001]` | `r_lin0 + [rc + disturb_xlin, 0, 0]`, `r_lin0 = [0.2861, 0, 0.001]` | sí (desde 2026-10-24; antes faltaba la perturbación) |

`distxlin` se llama `disturb_xlin` en el port (para no confundir "dist"
con distancia; por lo mismo, `dist_xg`, `dist_yg` → `disturb_xg`,
`disturb_yg`).

**Qué representa** (autor, 2026-10-24): que rc no queda donde el control
cree, es decir, el cero virtual del actuador está mal calculado. Como
entra en r_lin, pasa por Mbr+Ma fcn y afecta todo: M, I_b, r_g y la
restauración. A diferencia de `disturb_xg`/`disturb_yg`, que se suman a r_g
después de armar M.

Ejemplo de su efecto: `disturb_xlin = 5 mm` en la prueba de validación lleva
el pitch de bajada de −38.5° a −44.5° y el de subida de +36.6° a +29.1°.

### 2.6 Superficie: `fcn_buoyancy_surface`

Referencia: [referencia_simulink/fcn_buoyancy_surface.m](referencia_simulink/fcn_buoyancy_surface.m)
(revisado 2026-10-24)

Port exacto en `dynamics.py::surface_two_point`, con
`surface_model="simulink"`:

- Emersión de la parte superior del casco solo en nariz y cola, sin roll:
  `z_tope = z ∓ (L/2)·sin θ − (H/2)·cos θ`, fracción emergida
  `clip(−z_tope/H, 0, 1)`.
- `B_scale = 1 − promedio`, `Δx_b = −(L/2)·(e_nariz − e_cola)/(e_nariz + e_cola)·promedio`.
- Conexión: `B = B_total·B_scale` y `x_b = x_b(VBD) + Δx_b`, antes del
  vector de restauración. Confirmado por el autor (2026-10-24): `B_scale`
  multiplica a B y `Δx_b` se suma a x_b.
- `L = hull_L = 1.8`, `H = hull_H = 0.2` (`[superficie]` del archivo de
  configuración).

Lejos de la superficie no hace nada: verificado, la prueba de validación da
exactamente lo mismo con `"simulink"` que con `"none"`.

**Comparación con la superficie completa del port** (`"full"`): subida
desde 20 m con dB = +3 N hasta quedar flotando. Promedio de los últimos
50 s:

| Modelo | Llega a z < 0.5 m | z flotando | pitch | B_scale |
|---|---|---|---|---|
| `simulink` (2 puntos) | 66 s | +0.131 m | +2.1° | 0.995 |
| `full` (9 estaciones, roll, masa añadida, radiación) | 65 s | +0.137 m | +3.2° | 0.995 |

Los dos flotan con el 0.5 % del volumen fuera, que equivale a los +3 N de
flotabilidad (0.5 % de 569 N). La diferencia está en el transitorio de
llegada y el pitch final.

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

Switch `surface_model` (desde 2026-10-24):

| Valor | Qué es |
|---|---|
| `"none"` | sin superficie: cerca de z = 0 no es físico (con flotabilidad positiva sube hacia z negativo) |
| `"simulink"` | tu `fcn_buoyancy_surface` de 2 puntos (sección 2.6). Perfil `simulink` |
| `"full"` | modelo del port, descrito abajo. Perfil `completo` y default de `ModelConfig()` |

Modelo completo del port (`dynamics.py::surface_state, tau_surface`):

- Calcula qué fracción del casco está fuera del agua (9 puntos a lo largo,
  con pitch y roll).
- Escala la flotabilidad por la fracción sumergida y mueve el centro de
  flotación al centroide sumergido.
- Reduce la masa añadida al emerger (mínimo `ma_scale_floor = 0.3`).
- Agrega cerca de la superficie: fuerza de entrada al agua y amortiguamiento
  de radiación en heave y pitch.

**Efecto principal** (cualquiera de los dos modelos): con flotabilidad
positiva el planeador sube hasta quedar parcialmente fuera y se queda
flotando (en la misión, ~0.1 m en FIX). Sin modelo de superficie (v1 NO_WP)
subiría indefinidamente hacia z negativo. Comparación entre los dos en la
sección 2.6.

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

## 7. Switches del modelo

Se eligen como perfil en `config/cidesi.toml` (`[modelo.simulink]`,
`[modelo.completo]`; se pueden agregar otros). Cada perfil debe listar
todos los switches. La clase es `ModelConfig` en `params.py`; sus defaults
equivalen al perfil `completo` y solo los usan los scripts de v1.
Cambiar cualquiera obliga a recompilar (~5–10 s).

| Switch | Perfil `simulink` | Perfil `completo` | Qué hace |
|---|---|---|---|
| `kmr_mode` | `"signum"` | `"signum"` | `"signum"` = KMR·sign(α); `"parity"` = variante suave; `"const"` = sin dependencia de α |
| `kmy_cubic` | true | true | término cúbico en el momento de yaw |
| `use_alpha_saturation` | false | false | sección 3 |
| `use_rot_damping` | true | true | D_body |
| `surface_model` | `"simulink"` | `"full"` | sección 5 |
| `use_crossflow` | false | true | sección 4 |
| `use_geodesy` | false | true | posición ECEF (necesaria para la misión con GPS) |
| `use_ca_omega` | false | false | Coriolis de masa añadida (rotacional) |
| `use_ca_translational` | false | false | bloque de Munk: **duplica** física ya incluida en el CFD |
| `replicate_disturb_wiring` | true | true | la perturbación de CG (`disturb_xg`, `disturb_yg`) entra en restauración y C_RB pero no en M_RB, como en Simulink |
