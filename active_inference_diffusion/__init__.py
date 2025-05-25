"""
Active Inference + Diffusion package
"""

from .configs.config import (
    ActiveInferenceConfig,
    PixelObservationConfig,
    TrainingConfig,
    DiffusionConfig,
    BeliefDynamicsConfig
)

from .agents import (
    StateBasedAgent,
    PixelBasedAgent,
    BaseActiveInferenceAgent
)

from .core import (
    ActiveInferenceCore,
    BeliefDynamics,
    DiffusionProcess,
    FreeEnergyComputation
)

__version__ = "0.1.0"

__all__ = [
    # Configs
    "ActiveInferenceConfig",
    "PixelObservationConfig",
    "TrainingConfig",
    "DiffusionConfig",
    "BeliefDynamicsConfig",
    
    # Agents
    "StateBasedAgent",
    "PixelBasedAgent",
    "BaseActiveInferenceAgent",
    
    # Core
    "ActiveInferenceCore",
    "BeliefDynamics",
    "DiffusionProcess",
    "FreeEnergyComputation",
]