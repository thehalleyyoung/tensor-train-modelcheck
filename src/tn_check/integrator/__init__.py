"""
Time integration engine for CME on tensor-train format.

Implements multiple integration methods:
- TDVP (Time-Dependent Variational Principle) 1-site and 2-site
- Adaptive Euler forward
- Convergence analysis for non-Hermitian generators
"""

from tn_check.integrator.tdvp import (
    TDVPIntegrator,
    tdvp_one_site_sweep,
    tdvp_two_site_sweep,
)
from tn_check.integrator.base import (
    IntegratorBase,
    IntegrationResult,
    TimePoint,
)

__all__ = [
    "TDVPIntegrator", "tdvp_one_site_sweep", "tdvp_two_site_sweep",
    "IntegratorBase", "IntegrationResult", "TimePoint",
]

# Optional imports for modules that may not be present
try:
    from tn_check.integrator.krylov import (
        KrylovIntegrator,
        tt_krylov_step,
        lanczos_iteration,
    )
    __all__ += ["KrylovIntegrator", "tt_krylov_step", "lanczos_iteration"]
except ImportError:
    pass

try:
    from tn_check.integrator.uniformization import (
        UniformizationIntegrator,
        fox_glynn_weights,
        uniformization_step,
    )
    __all__ += ["UniformizationIntegrator", "fox_glynn_weights", "uniformization_step"]
except ImportError:
    pass

try:
    from tn_check.integrator.euler import (
        EulerIntegrator,
        euler_step,
        adaptive_euler_step,
    )
    __all__ += ["EulerIntegrator", "euler_step", "adaptive_euler_step"]
except ImportError:
    pass

try:
    from tn_check.integrator.steady_state import (
        SteadyStateSolver,
        dmrg_steady_state,
    )
    __all__ += ["SteadyStateSolver", "dmrg_steady_state"]
except ImportError:
    pass
