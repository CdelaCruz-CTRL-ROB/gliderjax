"""The author's waypoint FSM, port of fcn_Glider_State_Machine (Simulink,
version of 2026-10; source in docs/referencia_simulink/fsm_y_geodesia.m).
Names of constants and persistent variables are the Simulink ones.

Deliberate differences from the Simulink source:
  - lla2ecef uses sin(lon) for y (Simulink: cos(lon), an error that
    distorts the FSM's map, up to ~22 deg of heading; BITACORA entry 17).
  - DONE asks for positive buoyancy (buoy_done_ref = buoy_up_ref; Simulink
    leaves the default 0 = neutral). Author's decision, 2026-10-24.
  - The dive model (predict + plan a dive) is pluggable; the default,
    STRAIGHT_GLIDE, is the Simulink one. yaw_at_plan is kept for models
    that need the heading at the start of the dive (Simulink does not).
  - Optional pitch per leg (leg_pitch): the dives toward waypoint i use its
    own pitch_down / pitch_up (Simulink: one pitch for the whole mission;
    the author gives the pitch explicitly per dive, 2026-10-24). Without
    leg_pitch it is the Simulink FSM.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..navigation import (Navigator, nav_out, mission_geometry, fix_to_ned)
from .dive_models import STRAIGHT_GLIDE


class Config(NamedTuple):
    acceptance_radius_m: float = 90.0
    surface_wait_time_s: float = 60.0
    bottom_wait_time_s: float = 10.0
    pitch_antenna_rad: float = -1.1
    pitch_down_rad: float = -0.6109
    pitch_up_rad: float = 0.6109
    buoy_down_ref: float = 3.0
    buoy_up_ref: float = -3.0
    buoy_done_ref: float = -3.0          # Simulink: 0 (neutral); author: positive
    max_dive_depth_m: float = 200.0
    depth_epsilon_m: float = 2.0
    surface_epsilon_m: float = 1.0
    initial_turn_depth_m: float = 2.0
    flare_depth_m: float = 3.5
    surface_stabilize_time_s: float = 10.0
    min_estimation_depth_m: float = 15.0
    pitch_flare_target: float = -1.1
    flare_duration_s: float = 10.0
    baseline_glide_ratio: float = 1.4
    alpha_filter: float = 0.8


STATE_NAMES = ("FIX_PLAN", "SUBMERGE", "DIVE_TURN", "BOTTOM_WAIT", "CLIMB",
               "DONE", "SURFACE_WAIT")
(FIX_PLAN, SUBMERGE, DIVE_TURN, BOTTOM_WAIT, CLIMB, DONE,
 SURFACE_WAIT) = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


class Carry(NamedTuple):
    """Persistent variables. pitch_at_flare_entry is not stored: Simulink
    always sets it to pitch_up_rad (here, the leg's pitch_up)."""
    glider_state: float
    current_wp_index: float          # 1-based
    t_state_start: float
    current_yaw_ref: float
    yaw_ref_last_dive: float
    current_dive_depth_m: float
    dive_start_n: float
    dive_start_e: float
    max_depth_this_dive: float
    D_norm_n: float
    D_norm_e: float
    dist_to_wp_last_fix: float
    Dist_to_Target: float
    flare_active: float              # 0/1
    t_flare_start: float
    yaw_at_plan: float               # not in Simulink (see module doc)


def make(mission_lla, dive_model=STRAIGHT_GLIDE, leg_pitch=None, **options):
    """Navigator for this FSM. options: any Config field, e.g.
    make(mission, pitch_down_rad=-0.5). leg_pitch: optional, one
    (pitch_down_rad, pitch_up_rad) per waypoint, the pitch of the dives
    toward that waypoint; NaN = the Config value."""
    cfg = Config(**options)
    geo = mission_geometry(mission_lla)
    WPS = jnp.asarray(geo.waypoints_ned)
    NWP = WPS.shape[0]
    lp = np.full((NWP, 2), np.nan) if leg_pitch is None else np.asarray(leg_pitch, dtype=float)
    if lp.shape != (NWP, 2):
        raise ValueError(f"leg_pitch: one (pitch_down, pitch_up) per waypoint, shape ({NWP}, 2)")
    lp = np.where(np.isnan(lp), [cfg.pitch_down_rad, cfg.pitch_up_rad], lp)
    if (lp[:, 0] >= 0).any() or (lp[:, 1] <= 0).any():
        raise ValueError("leg_pitch: pitch_down must be < 0 and pitch_up > 0")
    LEG_PITCH = jnp.asarray(lp)

    def wp(index):                                  # 1-based, clamped
        return WPS[jnp.clip(index.astype(jnp.int32) - 1, 0, NWP - 1)]

    def leg(index):
        """(pitch_down, pitch_up) of the dives toward waypoint `index`."""
        return LEG_PITCH[jnp.clip(index.astype(jnp.int32) - 1, 0, NWP - 1)]

    def defaults(s):
        # outputs before the switch: pitch 0, current yaw, buoy 0
        return nav_out(0.0, s.yaw, 0.0, 0.0, 0.0)

    def fix_plan(c, s):
        out = nav_out(cfg.pitch_antenna_rad, c.current_yaw_ref,
                      cfg.buoy_up_ref, 0.0, 0.0)
        ready = (s.t - c.t_state_start) >= cfg.surface_wait_time_s
        cur = fix_to_ned(s.fix, geo)
        # drift estimator
        est = ready & (c.max_depth_this_dive > cfg.min_estimation_depth_m)
        expected = (jnp.array([c.dive_start_n, c.dive_start_e])
                    + dive_model.displacement(c.yaw_at_plan, c.yaw_ref_last_dive,
                                              c.max_depth_this_dive, cfg))
        D_old = jnp.array([c.D_norm_n, c.D_norm_e])
        D_est = (cur - expected) / jnp.maximum(c.max_depth_this_dive, 1e-6)
        D_norm = jnp.where(est, cfg.alpha_filter * D_est
                           + (1.0 - cfg.alpha_filter) * D_old, D_old)
        # arrival
        dist_old = jnp.linalg.norm(wp(c.current_wp_index) - cur)
        arrived = ready & (dist_old < cfg.acceptance_radius_m)
        index = c.current_wp_index + jnp.where(arrived, 1.0, 0.0)
        exhausted = index > NWP
        finished = ready & exhausted
        go = ready & ~exhausted
        # plan the next dive
        vec = wp(index) - cur
        dist = jnp.linalg.norm(vec)
        yaw_plan, depth_plan = dive_model.plan(vec, D_norm, s.yaw, cfg)
        c = c._replace(
            D_norm_n=D_norm[0], D_norm_e=D_norm[1],
            current_wp_index=index,
            dist_to_wp_last_fix=jnp.where(
                ready, jnp.where(finished, dist_old, dist), c.dist_to_wp_last_fix),
            Dist_to_Target=jnp.where(go, dist, c.Dist_to_Target),
            current_yaw_ref=jnp.where(go, yaw_plan, c.current_yaw_ref),
            yaw_ref_last_dive=jnp.where(go, yaw_plan, c.yaw_ref_last_dive),
            yaw_at_plan=jnp.where(go, s.yaw, c.yaw_at_plan),
            current_dive_depth_m=jnp.where(go, depth_plan, c.current_dive_depth_m),
            dive_start_n=jnp.where(go, cur[0], c.dive_start_n),
            dive_start_e=jnp.where(go, cur[1], c.dive_start_e),
            max_depth_this_dive=jnp.where(go, 0.0, c.max_depth_this_dive),
            glider_state=jnp.where(go, SUBMERGE, jnp.where(finished, DONE, FIX_PLAN)),
            t_state_start=jnp.where(go, s.t, c.t_state_start))
        # Simulink `return` when the waypoints run out: default outputs
        return c, jnp.where(finished, defaults(s), out)

    def submerge(c, s):
        c = c._replace(max_depth_this_dive=jnp.maximum(c.max_depth_this_dive, s.depth))
        go = s.depth > cfg.initial_turn_depth_m
        c = c._replace(glider_state=jnp.where(go, DIVE_TURN, SUBMERGE),
                       t_state_start=jnp.where(go, s.t, c.t_state_start))
        return c, nav_out(leg(c.current_wp_index)[0], s.yaw, cfg.buoy_down_ref,
                          0.0, 0.0)

    def dive_turn(c, s):
        diff = c.current_yaw_ref - s.yaw            # shortest way, persisted
        best_yaw_ref = s.yaw + jnp.arctan2(jnp.sin(diff), jnp.cos(diff))
        c = c._replace(current_yaw_ref=best_yaw_ref,
                       max_depth_this_dive=jnp.maximum(c.max_depth_this_dive, s.depth))
        go = (c.current_dive_depth_m - s.depth) < cfg.depth_epsilon_m
        c = c._replace(glider_state=jnp.where(go, BOTTOM_WAIT, DIVE_TURN),
                       t_state_start=jnp.where(go, s.t, c.t_state_start))
        return c, nav_out(leg(c.current_wp_index)[0], best_yaw_ref,
                          cfg.buoy_down_ref, 1.0, 0.0)

    def bottom_wait(c, s):
        c = c._replace(max_depth_this_dive=jnp.maximum(c.max_depth_this_dive, s.depth))
        go = (s.t - c.t_state_start) > cfg.bottom_wait_time_s
        c = c._replace(flare_active=jnp.where(go, 0.0, c.flare_active),
                       glider_state=jnp.where(go, CLIMB, BOTTOM_WAIT),
                       t_state_start=jnp.where(go, s.t, c.t_state_start))
        return c, nav_out(0.0, c.current_yaw_ref, 0.0, 0.0, 0.0)

    def climb(c, s):
        # flare: from flare_depth_m, pitch ramps in time from pitch_up
        # (pitch_at_flare_entry) to pitch_flare_target (antenna up) over
        # flare_duration_s
        pitch_up = leg(c.current_wp_index)[1]
        active = c.flare_active > 0.5
        progress = jnp.clip((s.t - c.t_flare_start) / cfg.flare_duration_s, 0.0, 1.0)
        pitch = jnp.where(active, pitch_up + (cfg.pitch_flare_target - pitch_up)
                          * progress, pitch_up)
        start = ~active & (s.depth <= cfg.flare_depth_m)
        c = c._replace(flare_active=jnp.where(start, 1.0, c.flare_active),
                       t_flare_start=jnp.where(start, s.t, c.t_flare_start))
        surf = s.depth <= cfg.surface_epsilon_m
        c = c._replace(flare_active=jnp.where(surf, 0.0, c.flare_active),
                       glider_state=jnp.where(surf, SURFACE_WAIT, CLIMB),
                       t_state_start=jnp.where(surf, s.t, c.t_state_start))
        return c, nav_out(pitch, c.current_yaw_ref, cfg.buoy_up_ref, -1.0, 0.0)

    def mission_done(c, s):
        return c, nav_out(cfg.pitch_antenna_rad, s.yaw, cfg.buoy_done_ref, 0.0, 1.0)

    def surface_wait(c, s):
        go = (s.t - c.t_state_start) >= cfg.surface_stabilize_time_s
        c = c._replace(glider_state=jnp.where(go, FIX_PLAN, SURFACE_WAIT),
                       t_state_start=jnp.where(go, s.t, c.t_state_start))
        return c, nav_out(cfg.pitch_antenna_rad, c.current_yaw_ref,
                          cfg.buoy_up_ref, 0.0, 0.0)

    def typed(branch):
        # lax.switch needs every branch to return the same types
        def f(c, s):
            c, out = branch(c, s)
            return Carry(*(jnp.asarray(v, dtype=float) for v in c)), out
        return f

    branches = [typed(b) for b in (fix_plan, submerge, dive_turn, bottom_wait,
                                   climb, mission_done, surface_wait)]

    def step(carry, s):
        c = Carry(*carry)
        i = jnp.clip(c.glider_state.astype(jnp.int32) - 1, 0, 6)
        c, out = jax.lax.switch(i, branches, c, s)
        return jnp.stack(c), out

    carry0 = jnp.asarray(Carry(
        glider_state=FIX_PLAN, current_wp_index=1.0, t_state_start=0.0,
        current_yaw_ref=0.0, yaw_ref_last_dive=0.0,
        current_dive_depth_m=cfg.max_dive_depth_m, dive_start_n=0.0,
        dive_start_e=0.0, max_depth_this_dive=0.0, D_norm_n=0.0, D_norm_e=0.0,
        dist_to_wp_last_fix=0.0, Dist_to_Target=0.0, flare_active=0.0,
        t_flare_start=0.0, yaw_at_plan=0.0), dtype=float)

    return Navigator(name=f"simulink ({dive_model.name})", carry0=carry0,
                     step=step, carry_names=Carry._fields,
                     state_names=STATE_NAMES, waypoints_ned=geo.waypoints_ned,
                     origin_lla_deg=geo.origin_lla_deg, ecef0=geo.ecef0,
                     config=cfg)
