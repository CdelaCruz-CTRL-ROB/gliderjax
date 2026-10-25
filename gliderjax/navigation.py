"""Navigation: what every state machine (FSM) shares. The FSMs themselves
live in gliderjax/fsm/, one file each (see gliderjax/fsm/__init__.py).

Contract of a FSM -- anything that follows it plugs into
autopilot.make_waypoint_controller:

  Navigator(name, carry0, step, carry_names, state_names,
            waypoints_ned, origin_lla_deg, ecef0)
  carry, out = step(carry, sens)
    carry  (n,) floats, the FSM's memory (Simulink: persistent variables);
           any length, names in carry_names (they show up in telemetry)
    sens   Sensors: what the FSM may read (no truth beyond the sensors)
    out    (5,) = NAV_OUT_NAMES: [pitch_ref [rad], yaw_ref [rad],
           buoy_ref [N, FSM convention: + = dive; autopilot.BUOY_SIGN maps it
           to the VBD], turn_sign (+1 dive, -1 climb, 0 = no heading control),
           mission_complete (0/1)]
  Must be JAX-traceable: jnp only, no python `if` on values.

Missions are lists of waypoints [lon_deg, lat_deg, h] (Fossen's (l, mu, h)
order, the same as the author's Simulink). The first waypoint is the origin
of the local NED frame.
"""
from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np

from .geodesy import lla2ecef, ecef2lla, waypoints_lla_to_ned


class Sensors(NamedTuple):
    """What a FSM can read each controller tick."""
    fix: jnp.ndarray       # (2,) [lon_deg, lat_deg], last GPS fix (held underwater)
    gps_ok: jnp.ndarray    # 1.0 when the fix is fresh (near the surface), else 0.0
    depth: jnp.ndarray     # [m] depth sensor (z, positive down)
    roll: jnp.ndarray      # [rad]
    pitch: jnp.ndarray     # [rad]
    yaw: jnp.ndarray       # [rad] heading (continuous, may leave +-pi)
    t: jnp.ndarray         # [s]


NAV_OUT_NAMES = ("pitch_ref", "yaw_ref", "buoy_ref", "turn_sign",
                 "mission_complete")


def nav_out(pitch_ref, yaw_ref, buoy_ref, turn_sign, mission_complete):
    """The FSM output vector, in NAV_OUT_NAMES order."""
    return jnp.array([pitch_ref, yaw_ref, buoy_ref, turn_sign,
                      mission_complete], dtype=float)


class Navigator(NamedTuple):
    name: str
    carry0: jnp.ndarray            # initial memory
    step: Callable                 # (carry, Sensors) -> (carry, out (5,))
    carry_names: tuple             # one name per carry entry
    state_names: tuple             # names of the discrete states (1-based)
    waypoints_ned: np.ndarray      # (n, 2) [N, E] from the origin [m]
    origin_lla_deg: np.ndarray     # (3,) first waypoint [lon_deg, lat_deg, h]
    ecef0: np.ndarray              # (3,) ECEF of the origin: start the sim here
    config: object = None          # the FSM's constants (for prints and plots)


# ---- missions ---------------------------------------------------------------
def check_mission(mission_lla):
    """(n, 3) array of [lon_deg, lat_deg, h]; raises if a value cannot be a
    longitude/latitude (catches [lat, lon] typed by mistake wherever
    |lon| > 90, e.g. the Gulf of Mexico: lat -95 is impossible)."""
    m = np.asarray(mission_lla, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("mission: one row per waypoint, [lon_deg, lat_deg, h]")
    if m.shape[1] == 2:
        m = np.column_stack([m, np.zeros(len(m))])
    bad = (np.abs(m[:, 1]) > 90.0) | (np.abs(m[:, 0]) > 180.0)
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f"mission: WP{i} (row {i + 1}; the origin is WP0) = "
                         f"{m[i, :2].tolist()} is not [lon_deg, lat_deg] "
                         "(|lat| <= 90). Swapped lat/lon?")
    return m[:, :3]


class MissionGeometry(NamedTuple):
    origin_lla_deg: np.ndarray     # (3,)
    ecef0: np.ndarray              # (3,) ECEF of the origin
    R_e_n: np.ndarray              # (3, 3) ECEF -> NED at the origin
    waypoints_ned: np.ndarray      # (n, 2) [N, E]


def mission_geometry(mission_lla):
    """Build-time (numpy): waypoints in the local NED frame of the origin."""
    m = check_mission(mission_lla)
    _, ecef0, R_e_n, ned = waypoints_lla_to_ned(m)
    return MissionGeometry(m[0], np.asarray(ecef0), np.asarray(R_e_n),
                           np.asarray(ned)[:, 0:2])


def fix_to_ned(fix_deg, geo: MissionGeometry):
    """[lon_deg, lat_deg] (h = 0) -> [N, E] from the mission origin (JAX).
    Same as the author's FSM: lla2ecef, minus the origin, rotated by R_e_n."""
    lla = jnp.array([jnp.deg2rad(fix_deg[0]), jnp.deg2rad(fix_deg[1]), 0.0])
    ned = jnp.asarray(geo.R_e_n) @ (lla2ecef(lla) - jnp.asarray(geo.ecef0))
    return ned[0:2]


# ---- GPS --------------------------------------------------------------------
def gps_fix(p, ecef_truth, depth, held_fix):
    """GPS sensor: ([lon_deg, lat_deg], ok). Fresh from the ECEF truth only
    near the surface (depth < gps_gate_depth), otherwise the last fix is held
    and ok = 0. (Noise via gps_sigma_m: TODO -- needs PRNG threading.)
    The author's Simulink has no GPS block: its FSM reads the true lat/lon
    all the time, but only uses it in FIX_PLAN, at the surface."""
    lla = ecef2lla(ecef_truth)
    fresh = jnp.array([jnp.rad2deg(lla[0]), jnp.rad2deg(lla[1])])
    ok = depth < p.gps_gate_depth
    return jnp.where(ok, fresh, held_fix), jnp.where(ok, 1.0, 0.0)


# ---- mission presets (from the V2 waypoint library) -------------------------
MISSION_TRIANGLE = [[-95.72275, 19.37214, 0.0],     # = the author's Simulink
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
