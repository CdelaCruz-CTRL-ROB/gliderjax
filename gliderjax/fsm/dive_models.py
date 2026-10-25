"""Dive models: the simplified model a FSM uses to predict where a dive
ends up and to plan the next one (heading and depth). Swapping it changes
the planning without touching the state machine.

A DiveModel has two functions (JAX, jnp only):

  displacement(yaw_start, yaw_ref, depth, cfg) -> [dN, dE]
      expected horizontal displacement of one complete dive (down to
      `depth` and back up) in still water. yaw_start = heading when the
      dive was planned, yaw_ref = planned heading. The drift estimator
      compares it with the real surfacing point.
  plan(vec, D_norm, yaw_now, cfg) -> (yaw_ref, dive_depth)
      vec = [N, E] from the glider to the target [m], D_norm = estimated
      drift per meter of depth [m/m], yaw_now = current heading.

cfg is the FSM's Config (glide ratio, depths, ...). A model that needs its
own parameters (turn rate, ...) is built by a function that takes them and
returns the DiveModel.
"""
from typing import Callable, NamedTuple

import jax.numpy as jnp


class DiveModel(NamedTuple):
    name: str
    displacement: Callable
    plan: Callable


# ---- straight glide: the author's Simulink FSM ------------------------------
def _straight_displacement(yaw_start, yaw_ref, depth, cfg):
    """Expected_Glide_Vector: 2 * depth * k along the planned heading."""
    return (2.0 * depth * cfg.baseline_glide_ratio) * jnp.array(
        [jnp.cos(yaw_ref), jnp.sin(yaw_ref)])


def _straight_plan(vec, D_norm, yaw_now, cfg):
    """Simulink FSM, STATE_FIX_PLAN, lines 288-313 of fsm_y_geodesia.m."""
    dist = jnp.linalg.norm(vec)
    horizontal_baseline = 2.0 * cfg.max_dive_depth_m * cfg.baseline_glide_ratio
    single_dive_drift = D_norm * cfg.max_dive_depth_m
    num_dives_needed = dist / (horizontal_baseline + 1e-6)
    required = vec - single_dive_drift * num_dives_needed
    yaw_ref = jnp.arctan2(required[1], required[0])
    depth = jnp.linalg.norm(required) / (2.0 * cfg.baseline_glide_ratio)
    depth = jnp.clip(depth, cfg.initial_turn_depth_m * 2.0, cfg.max_dive_depth_m)
    return yaw_ref, depth


STRAIGHT_GLIDE = DiveModel("recta", _straight_displacement, _straight_plan)

# by name, for the scripts (MODELO_INMERSION = "recta")
DIVE_MODELS = {m.name: m for m in (STRAIGHT_GLIDE,)}
