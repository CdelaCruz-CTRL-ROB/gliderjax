# gliderjax

Differentiable, GPU-batchable 6-DOF underwater glider simulator (JAX).
Port target: the CIDESI–ININDETEC Simulink model (flow-frame damping
formulation, "Model 3" of the project report).

**Status: working scaffold.** Dynamics structure, integration engine,
telemetry/scope infrastructure, closed-loop controller hook, batching and
validation harness are in place and tested with Seawing-like placeholder
parameters. Your job is transcription of the real parameters + validation
— not architecture.

```
gliderjax/params.py     GliderParams (pytree) + ModelConfig (static switches)
gliderjax/dynamics.py   kinematics, M_RB/C_RB(actuators), flow-frame hydro
                        (parity-structured), restoring, C_A blocks (ledger)
gliderjax/sim.py        RK4 + lax.scan, rate-limited actuators, discrete
                        controller hook, vmap batch runner
gliderjax/scope.py      Simulink-scope equivalent: plot ANY signal by name
demos/                  open-loop glide | closed-loop pitch PID | batch/MC
validation/             Simulink CSV comparison harness
```

## Quick start

```bash
pip install jax matplotlib
python demos/demo_openloop.py     # steady glide + full scope set in out/
python demos/demo_closedloop.py   # PID-form pitch controller, dive+climb
python demos/demo_batch.py        # dx family + 256-run Monte Carlo
```

## The scope system (your debugging lifeline)

Every call to `dynamics()` returns a telemetry dict containing **every
internal signal**: each force/moment contribution separately
(`tau_flow_body`, `tau_restoring`, `tau_drot`, `tau_coriolis_rb`, `tau_CA`),
body velocities/accelerations, flow angles, inertial pos/vel/accel, actuator
states and commands. The scan stacks them over time.

```python
tel = simulate(params, x0, n_frames)
scope.list_signals(tel)                                  # inventory
scope.scope(tel, ["pitch_deg", "alpha_deg", "nu_r:0"])   # ad-hoc scope
scope.moments_breakdown(tel, axis=4)                     # WHICH physics dominates
scope.default_scopes(tel, outdir="out")                  # standard debug set
```

`moments_breakdown` is the plot that finds sign errors and double-counted
terms in minutes — it found two during the scaffold's own bring-up.

## Porting checklist (Simulink -> here)

1. **Parameters** (`params.py`): replace every `# <-- REPLACE` field with the
   CIDESI values (mass budget, inertias, actuator geometry/limits, ANSYS
   coefficients, MA from your potential-flow source; MA_hull = hull-only).
   Keep the vehicle *trimmed*: x_g0 ~ 0 (the scaffold's first bug was an
   untrimmed placeholder budget — the scopes caught it immediately).
2. **Hydro model** (`dynamics.py::tau_flowframe`): the parity structure
   K(a) = K_e + K_o*CLn(a) is in place; put your constant-coefficient values
   in K_e with K_o = 0 to reproduce Simulink exactly, then populate K_o from
   the +-alpha CFD sweeps. Replace the tanh envelope guard with your
   crossflow model when available.
3. **Model variants** (`ModelConfig`): `use_ca_translational` exists ONLY to
   demonstrate the Munk double-count; keep False. `use_ca_omega` adds the
   omega-coupled block from MA_hull for channels not fitted by rotary CFD.
4. **Controllers**: write each as `cmds, state = ctrl(t, x, state, params)`.
   It runs at a fixed discrete rate (`ctrl_every`) — this IS the embedded
   implementation. Port pitch (done as demo), then roll/yaw cascade with the
   K_phi_psi coupling.
5. **Validate**: export the 4 canonical Simulink scenarios to CSV
   (see `validation/compare_simulink.py` docstring; use ode4 fixed-step at
   the same dt) and run the comparator until all states PASS.
6. **Then and only then**: FSM navigation layer (integer state in the scan
   carry, `jnp.where` transitions), GPU (`jax.config`... float32 forces +
   float64 kinematics), gradients (`jax.grad` + `jax.checkpoint` over scan
   segments).

## Performance (this scaffold, CPU, float64)

* single run: 400 s mission (8k RK4 steps) in ~0.17 s compiled
  -> a 30 km / 1e5 s mission in ~40 s on one core (vs hours in Simulink)
* batch: 256 x 300 s in ~5.5 s aggregate on CPU; GPU scales this to
  1e3-1e5 trajectories.

## Conventions

NED inertial, body z down (keel), eta=[x y z phi theta psi],
nu_r=[u v w p q r] RELATIVE velocity (constant current supported via
params.current_n). Restoring per the report's g(eta); actuators are states
with rate+travel limits, zero-order-held within RK4 steps (quasi-static
internal-mass assumption of the report).
