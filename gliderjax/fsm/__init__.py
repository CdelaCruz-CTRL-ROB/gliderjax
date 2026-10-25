"""State machines (FSMs) for waypoint missions, one file each. Every FSM
exposes make(mission_lla, **options) -> Navigator (contract in
gliderjax/navigation.py) and is registered in FSMS, so a script picks it
by name:

    from gliderjax import fsm
    nav = fsm.make("simulink", mission)

To make an alternative:
  - Same logic, another dive model (how a dive is predicted and planned,
    e.g. counting the turn): write a DiveModel in dive_models.py, add it to
    DIVE_MODELS there and pass it,
    fsm.make("simulink", mission, dive_model=DIVE_MODELS["my_name"]).
  - Other logic: copy simulink.py to a new file, change it, register it.
"""
from . import dive_models, simulink

FSMS = {
    "simulink": simulink.make,      # the author's FSM (Simulink, 2026-10)
}


def make(name, mission_lla, **options):
    if name not in FSMS:
        raise ValueError(f"FSM '{name}' unknown; options: {list(FSMS)}")
    return FSMS[name](mission_lla, **options)
