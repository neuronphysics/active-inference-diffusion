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
    DiffusionPixelAgent,
    DiffusionStateAgent,
    BaseActiveInferenceAgent
)

from .core import (
    DiffusionActiveInference,
    LatentDiffusionProcess,
    BeliefDynamics,
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
    "DiffusionPixelAgent",
    "DiffusionStateAgent",
    "BaseActiveInferenceAgent",
    
    # Core
    "DiffusionActiveInference",
    "LatentDiffusionProcess",
    "BeliefDynamics",
    "FreeEnergyComputation",
]