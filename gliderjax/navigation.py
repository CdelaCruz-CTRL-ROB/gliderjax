"""Navigation: V2 state machine (verbatim port), GPS sensor, mission presets.

FSM carry layout (14 floats):
  [0] state (1..7)      [1] wp_index (1-based)  [2] t_state_start
  [3] current_yaw_ref   [4] yaw_ref_last_dive   [5] Dist_to_Target
  [6] current_dive_depth[7] dive_start_n        [8] dive_start_e
  [9] max_depth_this_dive [10] Dnorm_n [11] Dnorm_e
  [12] mission_complete [13] dist_to_wp_last_fix
FSM outputs (5): [pitch_ref, yaw_ref, Buoy_ref, coupling_gain, complete]
Buoy_ref uses the V2 sign convention (+3 dive, -3 climb); the autopilot maps
it to the VBD's physical dB (see autopilot.py -- CONFIRM the Simulink sign).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .geodesy import lla2ecef, ecef2lla, waypoints_lla_to_ned

# ---- V2 constants (report version) -----------------------------------------
ACCEPT_R = 90.0
SURFACE_WAIT = 30.0
BOTTOM_WAIT = 10.0
PITCH_ANT = -1.1
PITCH_DN = -0.6109
PITCH_UP = 0.6109
BUOY_DN = 3.0
BUOY_UP = -3.0
MAX_DEPTH = 200.0
DEPTH_EPS = 4.0
SURF_EPS = 1.0
TURN_DEPTH = 2.0
FLARE_DEPTH = 1.0
STAB_TIME = 10.0
MIN_EST_DEPTH = 20.0
K_GLIDE = 1.4
ALPHA_F = 0.8

# ---- mission presets (from the V2 waypoint library) -------------------------
MISSION_TRIANGLE = [[-95.72275, 19.37214, 0.0],
                    [-95.72011, 19.37464, 0.0],
                    [-95.72275, 19.37714, 0.0]]
MISSION_SQUARE7 = MISSION_TRIANGLE + [[-95.71798, 19.37714, 0.0],
                                      [-95.71798, 19.37264, 0.0],
                                      [-95.72275, 19.37264, 0.0],
                                      [-95.72275, 19.37714, 0.0]]
MISSION_SQUARE_3KM = [[-95.72275, 19.37214, 0.0],
                      [-95.72275, 19.40014, 0.0],
                      [-95.69412, 19.40014, 0.0],
                      [-95.69412, 19.37214, 0.0],
                      [-95.70844, 19.38614, 0.0]]
MISSION_LAWNMOWER = [[-95.72275, 19.37214, 0.0],
                     [-95.72275, 19.39014, 0.0],
                     [-95.71321, 19.39014, 0.0],
                     [-95.71321, 19.37214, 0.0],
                     [-95.70367, 19.37214, 0.0],
                     [-95.70367, 19.39014, 0.0],
                     [-95.69412, 19.39014, 0.0]]


def gps_fix(p, ecef_truth, depth, held_fix):
    """GPS sensor: [lon_deg, lat_deg] from ECEF truth, available only near
    the surface (depth < gps_gate_depth); otherwise the last fix is held.
    (Noise via gps_sigma_m: TODO -- needs PRNG threading; 0 = ideal.)"""
    lla = ecef2lla(ecef_truth)
    fresh = jnp.array([jnp.rad2deg(lla[0]), jnp.rad2deg(lla[1])])
    ok = depth < p.gps_gate_depth
    return jnp.where(ok, fresh, held_fix)


def build_fsm(mission_lla=MISSION_TRIANGLE):
    """Returns (carry0 (14,), ecef0 (3,), fsm_step).
    fsm_step(carry, lla_deg[2], depth, yaw, t) -> (carry', outputs[5])."""
    _, ecef0, _, wps_ned3 = waypoints_lla_to_ned(mission_lla)
    WPS = jnp.asarray(wps_ned3[:, 0:2])          # (n,2) NED
    NWP = WPS.shape[0]
    origin_lla, ecef_org, Ren, _ = waypoints_lla_to_ned(mission_lla)
    ECEF_ORG = jnp.asarray(ecef_org)
    REN = jnp.asarray(Ren)

    def lla_deg_to_ned(lla_deg):
        lla = jnp.array([jnp.deg2rad(lla_deg[0]), jnp.deg2rad(lla_deg[1]), 0.0])
        d = lla2ecef(lla) - ECEF_ORG
        ned = REN @ d
        return ned[0], ned[1]

    def wp(idx1):                                 # 1-based, clamped
        i = jnp.clip(idx1.astype(jnp.int32) - 1, 0, NWP - 1)
        return WPS[i]

    def out(pitch, yawr, buoy, cg, done):
        return jnp.array([pitch, yawr, buoy, cg, done])

    # ---- branches: (carry, lla, depth, yaw, t) -> (carry', out5) -----------
    def s_fix_plan(c, lla, depth, yaw, t):
        tis = t - c[2]
        ready = tis >= SURFACE_WAIT
        cn, ce = lla_deg_to_ned(lla)
        cur = jnp.array([cn, ce])
        # disturbance estimator (V2 lines 225-243)
        est_ok = c[9] > MIN_EST_DEPTH
        exp_vec = (2.0 * c[9] * K_GLIDE) * jnp.array([jnp.cos(c[4]), jnp.sin(c[4])])
        Dn_est = (cur - (c[7:9] + exp_vec)) / jnp.maximum(c[9], 1e-6)
        Dn_new = ALPHA_F * Dn_est + (1 - ALPHA_F) * c[10:12]
        Dn = jnp.where(ready & est_ok, Dn_new, c[10:12])
        # arrival check
        d0 = jnp.linalg.norm(wp(c[1]) - cur)
        arrived = ready & (d0 < ACCEPT_R)
        wpi = jnp.where(arrived, c[1] + 1.0, c[1])
        exhausted = wpi > NWP
        tgt = wp(wpi)
        # plan (V2 lines 292-329; note the depth-guess cancellation)
        vec = tgt - cur
        dist = jnp.linalg.norm(vec)
        drift = Dn * dist / (2.0 * K_GLIDE)       # = Dnorm*Dist/(2k), exact
        req = vec - drift
        yaw_plan = jnp.arctan2(req[1], req[0])
        depth_plan = jnp.clip(jnp.linalg.norm(req) / (2.0 * K_GLIDE),
                              TURN_DEPTH * 2.0, MAX_DEPTH)
        go = ready & (~exhausted)
        c = c.at[10:12].set(Dn)
        c = c.at[1].set(wpi)
        c = c.at[13].set(jnp.where(ready, dist, c[13]))
        c = c.at[5].set(jnp.where(ready, dist, c[5]))
        c = c.at[3].set(jnp.where(go, yaw_plan, c[3]))
        c = c.at[4].set(jnp.where(go, yaw_plan, c[4]))
        c = c.at[6].set(jnp.where(go, depth_plan, c[6]))
        c = c.at[7].set(jnp.where(go, cn, c[7]))
        c = c.at[8].set(jnp.where(go, ce, c[8]))
        c = c.at[9].set(jnp.where(go, 0.0, c[9]))
        c = c.at[0].set(jnp.where(go, 2.0, jnp.where(ready & exhausted, 6.0, 1.0)))
        c = c.at[2].set(jnp.where(go, t, c[2]))
        # V2 'return' quirk: on the exhausted tick, outputs are the defaults
        o_norm = out(PITCH_ANT, c[3], BUOY_UP, 0.0, 0.0)
        o_exh = out(0.0, yaw, 0.0, 0.0, 0.0)
        return c, jnp.where(ready & exhausted, o_exh, o_norm)

    def s_submerge(c, lla, depth, yaw, t):
        c = c.at[9].set(jnp.maximum(c[9], depth))
        go = depth > TURN_DEPTH
        c = c.at[0].set(jnp.where(go, 3.0, 2.0))
        c = c.at[2].set(jnp.where(go, t, c[2]))
        return c, out(PITCH_DN, yaw, BUOY_DN, 0.0, 0.0)

    def s_dive_turn(c, lla, depth, yaw, t):
        # yaw unwrap (shortest path, persisted -- V2 lines 236-239 semantics)
        cyr = yaw + jnp.arctan2(jnp.sin(c[3] - yaw), jnp.cos(c[3] - yaw))
        c = c.at[3].set(cyr)
        c = c.at[9].set(jnp.maximum(c[9], depth))
        go = (c[6] - depth) < DEPTH_EPS
        c = c.at[0].set(jnp.where(go, 4.0, 3.0))
        c = c.at[2].set(jnp.where(go, t, c[2]))
        return c, out(PITCH_DN, cyr, BUOY_DN, 1.0, 0.0)

    def s_bottom_wait(c, lla, depth, yaw, t):
        c = c.at[9].set(jnp.maximum(c[9], depth))
        go = (t - c[2]) > BOTTOM_WAIT
        c = c.at[0].set(jnp.where(go, 5.0, 4.0))
        c = c.at[2].set(jnp.where(go, t, c[2]))
        return c, out(0.0, c[3], 0.0, 0.0, 0.0)

    def s_climb(c, lla, depth, yaw, t):
        prog = jnp.clip((FLARE_DEPTH - depth) / (FLARE_DEPTH - SURF_EPS), 0.0, 1.0)
        pitch = jnp.where(depth > FLARE_DEPTH, PITCH_UP, PITCH_UP * (1.0 - prog))
        go = depth <= SURF_EPS
        c = c.at[0].set(jnp.where(go, 7.0, 5.0))
        c = c.at[2].set(jnp.where(go, t, c[2]))
        return c, out(pitch, c[3], BUOY_UP, -1.0, 0.0)

    def s_done(c, lla, depth, yaw, t):
        c = c.at[12].set(1.0)
        return c, out(PITCH_ANT, yaw, 0.0, 0.0, 1.0)

    def s_surface_wait(c, lla, depth, yaw, t):
        go = (t - c[2]) >= STAB_TIME
        c = c.at[0].set(jnp.where(go, 1.0, 7.0))
        c = c.at[2].set(jnp.where(go, t, c[2]))
        return c, out(0.0, c[3], BUOY_UP, 0.0, 0.0)

    branches = [s_fix_plan, s_submerge, s_dive_turn, s_bottom_wait,
                s_climb, s_done, s_surface_wait]

    def fsm_step(carry, lla_deg, depth, yaw, t):
        idx = jnp.clip(carry[0].astype(jnp.int32) - 1, 0, 6)
        return jax.lax.switch(idx, branches, carry, lla_deg, depth, yaw, t)

    carry0 = jnp.zeros(14).at[0].set(1.0).at[1].set(1.0).at[6].set(MAX_DEPTH)
    return carry0, jnp.asarray(ecef0), fsm_step
