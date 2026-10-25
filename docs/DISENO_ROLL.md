# Diseño del control de roll: signo del error y variable de control

Creado el 2026-10-24. **Pendiente**, para trabajarlo en otra sesión. El
simulador funciona tal como está hoy; este documento solo plantea el tema
de diseño.

## 1. Resumen

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

## 4. Lo que está implementado hoy

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

En el simulador (`gliderjax/autopilot.py::roll_control`, línea 103;
ganancias en las líneas 49-51):

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

Como referencia, el PID de pitch del simulador (línea 85), que el autor
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

1. **¿Cuál es la entrada de roll en el diseño del autor?** γ, sin γ, y_g
   u otra.
   - Si es γ, sin γ o "−y_g": el diseño está bien. Se implementa con esa
     variable (sección 5) y se declara el cambio de variable.
   - Si el diseño escribe φ̈ = … + (W/I)·y_g con e = φ − φ_ref y llega a
     y_g = +K·e: hay un error de signo en el diseño (es la variante que
     diverge).
2. ¿Se agrega a roll la compensación de gravedad, como en pitch?
   u_φ = (1/cos φ)·(KP·e + KD·ė + I − z_g·sin φ). Hoy el integrador la
   suple. Puede importar para la demostración.
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

## 7. Cómo probar

- `escenarios/02_simulacion.py`: `MODO = "pitch_roll"`, tabla
  `FASES_PITCH_ROLL`, `T_FINAL` ≥ 260 s.
- La figura `09_control.png` muestra roll contra la referencia, γ (comando
  y posición) y `mem[0]` de roll con ±`I_MAX_R`.
- Contexto: [CONTROL_Y_MISION.md](CONTROL_Y_MISION.md), secciones 2 y 5;
  [BITACORA.md](BITACORA.md), entradas 12 y 13.
