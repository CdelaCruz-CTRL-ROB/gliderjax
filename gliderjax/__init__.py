"""gliderjax: differentiable, GPU-batchable 6-DOF underwater glider simulator."""
import jax
jax.config.update("jax_enable_x64", True)  # float64 for long-mission accuracy

from .params import GliderParams, ModelConfig, seawing_like_defaults
from .dynamics import dynamics
from .sim import (make_simulator, make_resumable_simulator,
                  make_batch_simulator, initial_state,
                  build_initial_state, actuator_steady_state,
                  make_ref_pipeline, SOLVERS)
from . import scope
from . import geodesy, navigation, fsm, autopilot
