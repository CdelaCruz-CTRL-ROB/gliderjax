# Diseño del control de roll: signo del error y variable de control

Creado el 2026-10-24. **Pendiente**, para trabajarlo en otra sesión.

> **Actualización (mismo día):** el autor pasó su ley de roll completa. La
> pregunta 1 quedó resuelta y la parte de PID no lineal con compensación de
> gravedad ya está implementada. Ver la sección 8. Las secciones 1 a 5
> describen el estado anterior (PID simplificado de Simulink), y se dejan
> como contexto.

## 1. Resumen (estado anterior)

- El PID de roll del autor ya está en el simulador y funciona: llega a la
  referencia sin error en estado estacionario.
- Está escrito como en Simulink: **e = referencia − medido**, con salida
  y_g y γ = asin(−y_g·m_tot/(m_rot·R_rot)).
- En su diseño de control, el autor usa **e = medido − referencia** (como
  en pitch). Con esa convención, ganancias positivas y salida y_g, el lazo
  **diverge**.
- Si se toma como control **u_φ = −y_g**, el roll queda con la misma forma
  que pitch, y e = medido − referencia funciona. Es el mismo lazo que
  Simulink, bit a bit.
- Falta decidir: ¿cuál es la variable de entrada en el diseño del autor?
  ¿Hay que corregir el diseño o solo la implementación?

## 2. Convenciones del simulador

- Marco del cuerpo: x adelante, y estribor, z abajo. NED en el inercial.
- φ > 0: ala derecha (estribor) abajo. θ > 0: nariz arriba.
- p, q, r son velocidades angulares **del cuerpo**. Las derivadas de los
  ángulos salen de η̇ = J(η)ν (`autopilot.euler_rates(x)`):
  - φ̇ = p + (q·sin φ + r·cos φ)·tan θ
  - θ̇ = q·cos φ − r·sin φ
  - **No confundir p con φ̇ ni q con θ̇.**
- Masa rotatoria: r_rot = [x_rot, −R_rot·sin γ, R_rot·cos γ]
  (`dynamics.py::mass_geometry`, línea 90). γ > 0 la mueve a babor (y < 0).
- Centro de flotación en r_b = [x_b, 0, 0] (`dynamics.py::tau_restoring`,
  línea 286). z_b = 0, así que BG = z_g ≈ 6.93 mm.
- Valores CIDESI: m_tot = 58 kg, m_rot = 5 kg, R_rot = 0.07 m. Autoridad
  lateral máxima: |y_g| ≤ m_rot·R_rot/m_tot = 6.0 mm.

## 3. De dónde sale el signo

Momento del peso, r_g × f_g, con f_g = W·[−sin θ, sin φ cos θ, cos φ cos θ]
en el cuerpo. La flotación no da momento de roll porque y_b = z_b = 0.

```
pitch:  M_y = −W·(z_g·sin θ + x_g·cos θ·cos φ)
roll:   M_x = +W·cos θ·(y_g·cos φ − z_g·sin φ)
```

- Mover el CG hacia adelante (x_g > 0) produce pitch **negativo**: empuja
  contra θ.
- Mover el CG a estribor (y_g > 0) produce roll **positivo**: empuja a
  favor de φ.

Es el producto cruz con z hacia abajo. Por eso, con y_g como control, la
planta de roll tiene ganancia de signo contrario a la de pitch.

**Con u_φ = −y_g** (CG hacia babor, la dirección en que va la masa con
γ > 0):

```
roll:   M_x = −W·cos θ·(z_g·sin φ + u_φ·cos φ)
sin γ = u_φ · m_tot / (m_rot·R_rot)
```

Es exactamente la forma de pitch (con un factor cos θ), y sin γ ya no
lleva el signo menos.

## 4. Lo que estaba implementado (PID simplificado de Simulink)

PID del autor en Simulink:

```
y_g_control = 0.008*e + 0*0.01*ed + 0.01*ei
```

```matlab
function gamma = fcn(yg)
%   yg      : Posición lateral deseada del CG total [m]
  m_total=58;
  m_rot=5;
  r_rot=0.07;
    sin_gamma_val=- (yg * m_total)/(m_rot * r_rot);
    if sin_gamma_val>1
        sin_gamma_val=1;
    elseif sin_gamma_val<-1
        sin_gamma_val=-1;
    end
    gamma = asin(sin_gamma_val);
end
```

En el simulador (`gliderjax/autopilot.py::roll_control`, versión del commit
f68a962):

```python
phi = x[3]
phi_dot = euler_rates(x)[0]          # NO p
e = roll_ref - phi                   # referencia − medido (como Simulink)
de = roll_ref_dot - phi_dot
integ = jnp.clip(mem[0] + KI_R * e * dt, -I_MAX_R, I_MAX_R)
yg_cmd = KP_R * e + KD_R * de + integ
m_tot = params.m_static + params.m_lin + params.m_rot + params.m_pist
sin_gamma = jnp.clip(-yg_cmd * m_tot / (params.m_rot * params.R_rot), -1.0, 1.0)
return jnp.arcsin(sin_gamma), mem.at[0].set(integ)
```

- KP_R = 0.008, KD_R = 0 (en Simulink `0*0.01`: apagada), KI_R = 0.01.
- La integral guarda el **aporte** KI·∫e con tope ±I_MAX_R (provisional,
  4 mm). Es la regla del autor: KI fija la rapidez, I_MAX el máximo.
- m_tot no incluye la masa de perturbación `m_dist`.

En Simulink los dos signos menos (referencia − medido y el −y_g de la
`fcn`) se cancelan: sin γ = K·(φ − φ_ref)·m_tot/(m_rot·R_rot).

Como referencia, el PID de pitch del simulador (`pitch_control`), que el autor
considera equivalente a su PID no lineal:

```
e      = θ − θ_ref                       (medido − referencia)
ė      = θ̇ − θ̇_ref
I      = sat(I + KI·e·dt, ±I_MAX)
x_g    = (1/cos θ) · (KP·e + KD·ė + I − z_g·sin θ)
rc_cmd = m_tot · x_g / m_lin
```

## 5. Pruebas (2026-10-24)

Arrancando del reposo a 200 m con los actuadores en neutro. Bajada a −25°
con dB = −3 N, y escalón de roll 0 → 15° en t = 200 s (modo `pitch_roll`):

| Variante | Resultado |
|---|---|
| Implementado (e = ref − medido, salida y_g) | roll 15.000°; sobrepaso a 23.3°, periodo ~3 s; dentro de ±0.5° desde t = 207 s |
| e = medido − ref, salida y_g, mismas ganancias | **diverge**: roll −85°, γ pegado en +90° |
| e = medido − ref, salida u_φ = −y_g, sin γ = +u_φ·m/(m_rot·R) | **idéntico al implementado** (diferencia 0.0 en estado y comandos); `mem[0]` cambia de signo |
| `I_MAX_R` = 1 mm | roll se queda en 11.9° (el tope no alcanza) |

- Aporte integral: sube a 3.9 mm (casi el tope de 4) durante la subida del
  roll. En equilibrio vale 1.8 mm ≈ z_g·tan 15°: hace el papel de la
  compensación de gravedad que tiene pitch y roll no.
- KI/KP = 1.25 s⁻¹: la integral es rápida comparada con la respuesta, y
  explica buena parte del sobrepaso.
- Modo `rumbo` (0 → 45°): pasa 44° en t = 249 s; roll máximo 26.4°.

Código de la variante con u_φ = −y_g (para pegar en lugar del cuerpo de
`roll_control`):

```python
phi = x[3]
phi_dot = euler_rates(x)[0]
e = phi - roll_ref                       # medido − referencia
de = phi_dot - roll_ref_dot
integ = jnp.clip(mem[0] + KI_R * e * dt, -I_MAX_R, I_MAX_R)
u = KP_R * e + KD_R * de + integ         # u = −y_g
m_tot = params.m_static + params.m_lin + params.m_rot + params.m_pist
sin_gamma = jnp.clip(u * m_tot / (params.m_rot * params.R_rot), -1.0, 1.0)
return jnp.arcsin(sin_gamma), mem.at[0].set(integ)
```

## 6. Preguntas para la sesión de diseño

1. ~~¿Cuál es la entrada de roll en el diseño del autor?~~ **Resuelta**
   (sección 8): el y_g de la ley es el desplazamiento hacia babor,
   u = −y_g del marco del cuerpo. El diseño es consistente.
2. ~~¿Se agrega a roll la compensación de gravedad?~~ **Sí**, es parte de
   la ley del autor; ya está implementada (sección 8).
3. Factor cos θ en el momento de roll: con θ = ±25° la ganancia efectiva
   baja ~9 %. ¿Se compensa?
4. Parte D: con KD encendida y una referencia en escalón (modo
   `pitch_roll`, roll sin moldeado), φ̇_ref da un pico de 1 rad/s en una
   muestra. ¿Derivada sobre el error o solo sobre la medición? ¿Moldear la
   referencia de roll? (`make_ref_pipeline(bypass_roll=False)`)
5. Sintonización: sobrepaso de 55 % con las ganancias de Simulink. ¿Es lo
   esperado? ¿El periodo de control de Simulink es también 0.25 s?
6. Validar contra una corrida de Simulink exportada a CSV (ver
   [VALIDACION.md](VALIDACION.md)).

7. Faltan los términos −(d_φ/W)·φ̇, −(m_φ/W)·φ̈_d y Σ_ψ de la ley
   completa (sección 8). ¿Qué son d_φ, m_φ y Σ_ψ en el modelo del
   simulador? φ̈_d necesita una referencia de roll suave (hoy el modo
   `pitch_roll` la da en escalón).
8. Con la compensación de gravedad, las ganancias de Simulink dan más
   sobrepaso (sección 8): hay que sintonizar de nuevo.
9. z_g depende de γ (la masa rotatoria sube al girar: z_rot = R·cos γ).
   El controlador usa z_g con γ = 0; con γ = 17° la diferencia es 0.27 mm
   de 6.93 (4 %). ¿Importa?

## 7. Cómo probar

- `escenarios/02_simulacion.py`: `MODO = "pitch_roll"`, tabla
  `FASES_PITCH_ROLL`, `T_FINAL` ≥ 260 s.
- La figura `09_control.png` muestra roll contra la referencia, γ (comando
  y posición) y `mem[0]` de roll con ±`I_MAX_R`.
- Contexto: [CONTROL_Y_MISION.md](CONTROL_Y_MISION.md), secciones 2 y 5;
  [BITACORA.md](BITACORA.md), entradas 12 a 14.

## 8. Ley del autor (2026-10-24)

Ley completa:

```latex
y_g = \frac{1}{\cos(\phi)} \left[K_{P,\phi} e_\phi + K_{D,\phi} \dot{e}_\phi
      + K_{I,\phi} \int e_\phi - \frac{d_\phi}{W}\dot{\phi} - z_g \sin(\phi)
      - \frac{m_\phi}{W}\ddot{\phi}_d + \Sigma_{\psi} \right]
```

Implementada por ahora solo la parte de **PID no lineal con compensación
de gravedad**:

```
e  = φ − φ_ref                        (medido − referencia, igual que pitch)
ė  = φ̇ − φ̇_ref                        (φ̇ de la cinemática, NO p)
I  = sat(I + KI·e·dt, ±I_MAX)
u  = (1/cos φ) · (KP·e + KD·ė + I − z_g·sin φ)       ("y_g" de la ley)
sin γ = sat(u · m_tot / (m_rot·R_rot), ±1)
```

### El signo de y_g en la ley

El término de gravedad lo decide. Con e = 0 e I = 0 la ley da
y_g = −z_g·tan φ. Pero para sostener un roll φ > 0 el CG tiene que ir a
**estribor**, y_g(cuerpo) = +z_g·tan φ (sección 3: M_x = 0 ⇔
y_g·cos φ = z_g·sin φ). Entonces el y_g de la ley apunta a **babor**: es
u = −y_g del marco del cuerpo.

Con esa lectura, el roll tiene la misma forma que pitch,
M_x = −W·cos θ·(z_g·sin φ + u·cos φ), y la ley cancela la gravedad. Por
eso sin γ lleva signo **+** (la masa con γ > 0 va a babor). La `fcn` de
Simulink lleva signo − porque allí y_g es el de estribor.

Prueba: tomar la u de la ley como y_g de estribor (sin γ con signo −)
diverge (roll −85°, γ en +90°). La compensación de gravedad, en lugar de
cancelarse, se duplica.

### Resultados con las ganancias de Simulink (KP 0.008, KD 0, KI 0.01, I_MAX 4 mm)

Mismas pruebas que la sección 5:

| Prueba | PID simplificado (Simulink) | Ley con compensación |
|---|---|---|
| Escalón 0 → 15°, bajada | pico 23.3°, ±0.5° desde t = 207 s | pico **35.6°**, ±0.5° desde t = 239 s |
| Escalón 0 → −15°, subida | pico −23.3° | pico −35.6° |
| Aporte integral, final | 1.8 mm (hace de compensación de gravedad) | 0.06 mm (ya no hace falta) |
| Aporte integral, máximo | 3.9 mm | 4.0 mm (topado) |
| Rumbo 0 → 45° | pasa 44° en t = 249 s; roll máx 26.4° | pasa 44° en t = 249 s; roll máx 27.7° |

**Por qué empeora con las mismas ganancias:** el PID simplificado tenía
"gratis" la rigidez natural del péndulo, W·z_g (z_g = 6.9 mm), sumada a
KP = 8 mm. La ley la cancela, y queda solo KP: la rigidez baja a la mitad
y la integral (igual de rápida) domina la respuesta.

Algunas combinaciones (escalón de 15°, bajada), solo como orientación para
la sintonización:

| KP | KD | KI | Pico | ±0.5° desde | Aporte integral máx. |
|---|---|---|---|---|---|
| 0.008 | 0 | 0.01 | 35.6° | t = 239 s | 4.0 mm (topado) |
| 0.015 | 0 | 0.01 | 33.0° | t = 213 s | 3.9 mm |
| 0.015 | 0 | 0.003 | 21.1° | t = 211 s | 1.2 mm |
| 0.015 | 0.01 | 0.003 | 19.5° | t = 212 s | 1.2 mm |
| 0.008 | 0.01 | 0.01 | 28.2° | t = 227 s | 3.9 mm |

Aun con KD el sobrepaso no baja de ~30 %. Candidatos a revisar: la razón
de cambio de la masa rotatoria (0.5 rad/s: llevar γ a −40° tarda ~1.4 s),
el periodo de control de 0.25 s contra un roll con periodo de ~3 s, y la
referencia en escalón.
