"""Validation against Simulink reference runs.

WORKFLOW
--------
1. In Simulink, for each canonical scenario, log [t, eta(6), nu(6)] with a
   `To Workspace` block (Structure with time), then export:

     T = out.logsout;  % or your struct
     M = [t, x, y, z, phi, theta, psi, u, v, w, p, q, r];
     writematrix(M, "scenario_openloop.csv");

2. Run the same scenario here (same initial state, same actuator commands,
   fixed-step: set the Simulink solver to ode4 with the SAME dt for a clean
   comparison -- variable-step vs RK4 differences will otherwise dominate).

3. python validation/compare_simulink.py scenario_openloop.csv

Suggested canonical set (mirrors the report's figures):
  A. open-loop descent (dx=30mm, dB=-2.5N)          -> Fig 6.5 case
  B. open-loop roll (gamma step, neutral buoyancy)  -> Fig 6.4 case
  C. closed-loop pitch mission (two dives)          -> Fig 6.10 case
  D. spiral (gamma=45deg, dB=-3N)                   -> Fig 6.28 case

PASS criteria (tune to taste): RMS over the run below
  position 0.5 m | attitude 0.5 deg | velocity 0.005 m/s | rates 0.05 deg/s
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

TOL = {"pos": 0.5, "att_deg": 0.5, "vel": 5e-3, "rate_degs": 5e-2}


def compare(csv_path, tel):
    ref = np.loadtxt(csv_path, delimiter=",")
    t_ref = ref[:, 0]
    t = np.asarray(tel["t"])
    eta = np.asarray(tel["eta"]); nu = np.asarray(tel["nu_r"])
    # interpolate ours onto reference times
    def interp(col):
        return np.interp(t_ref, t, col)
    ours = np.column_stack([interp(eta[:, i]) for i in range(6)] +
                           [interp(nu[:, i]) for i in range(6)])
    err = ours - ref[:, 1:13]
    rms = np.sqrt(np.mean(err ** 2, axis=0))
    names = ["x", "y", "z", "phi", "theta", "psi",
             "u", "v", "w", "p", "q", "r"]
    ok = True
    for i, n in enumerate(names):
        lim = (TOL["pos"] if i < 3 else
               np.deg2rad(TOL["att_deg"]) if i < 6 else
               TOL["vel"] if i < 9 else np.deg2rad(TOL["rate_degs"]))
        flag = "PASS" if rms[i] < lim else "FAIL"
        ok &= rms[i] < lim
        print(f"  {n:6s} RMS {rms[i]:.3e}  (tol {lim:.1e})  {flag}")
    print("VALIDATION", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    print(__doc__)
