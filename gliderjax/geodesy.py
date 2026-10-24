"""WGS-84 geodesy (JAX, branchless). Conventions follow the Simulink blocks:
LLA vectors are [lon, lat, h] (radians, radians, meters); R_n_e maps NED -> ECEF
at the given (lon, lat); NED-from-origin = R_n_e(origin)^T @ (ECEF - ECEF_origin).
"""
import jax.numpy as jnp

R_E = 6378137.0
E_SQ = 0.00669437999014


def lla2ecef(lla):
    """[lon, lat, h] (rad, rad, m) -> ECEF xyz. (Corrected: y uses sin_lon.)"""
    lon, lat, h = lla[0], lla[1], lla[2]
    sl, cl = jnp.sin(lat), jnp.cos(lat)
    N = R_E / jnp.sqrt(1.0 - E_SQ * sl * sl)
    return jnp.array([(N + h) * cl * jnp.cos(lon),
                      (N + h) * cl * jnp.sin(lon),
                      (N * (1.0 - E_SQ) + h) * sl])


def ecef2lla(ecef, iters=4):
    """ECEF -> [lon, lat, h]. Fixed-count Bowring-type iteration (branchless;
    converges to <1e-9 rad at sea level in 3 iterations)."""
    x, y, z = ecef[0], ecef[1], ecef[2]
    lon = jnp.arctan2(y, x)
    p = jnp.sqrt(x * x + y * y)
    lat = jnp.arctan(z / (p * (1.0 - E_SQ)))
    h = 0.0
    N = R_E
    for _ in range(iters):
        sl = jnp.sin(lat)
        N = R_E / jnp.sqrt(1.0 - E_SQ * sl * sl)
        h = p / jnp.cos(lat) - N
        lat = jnp.arctan(z / (p * (1.0 - E_SQ * (N / (N + h)))))
    h = p / jnp.cos(lat) - N
    return jnp.array([lon, lat, h])


def R_n_e(lon, lat):
    """NED -> ECEF rotation at (lon, lat). Columns are ECEF directions of N,E,D."""
    cl, sl = jnp.cos(lon), jnp.sin(lon)
    cm, sm = jnp.cos(lat), jnp.sin(lat)
    return jnp.array([[-cl * sm, -sl, -cl * cm],
                      [-sl * sm,  cl, -sl * cm],
                      [ cm,      0.0, -sm]])


def waypoints_lla_to_ned(waypoints_lla_deg):
    """Mission-build-time conversion (plain math, called outside JIT).
    waypoints_lla_deg: (n,3) [lon_deg, lat_deg, h]. Returns (origin_lla_rad,
    ecef_origin, R_e_n_origin, waypoints_ned (n,3))."""
    import numpy as np
    wl = np.asarray(waypoints_lla_deg, dtype=float)
    origin = np.array([np.deg2rad(wl[0, 0]), np.deg2rad(wl[0, 1]), wl[0, 2]])
    ecef0 = np.asarray(lla2ecef(jnp.asarray(origin)))
    Ren = np.asarray(R_n_e(origin[0], origin[1])).T
    ned = np.zeros((wl.shape[0], 3))
    for i in range(wl.shape[0]):
        lla = np.array([np.deg2rad(wl[i, 0]), np.deg2rad(wl[i, 1]), wl[i, 2]])
        ned[i] = Ren @ (np.asarray(lla2ecef(jnp.asarray(lla))) - ecef0)
    return origin, ecef0, Ren, ned
