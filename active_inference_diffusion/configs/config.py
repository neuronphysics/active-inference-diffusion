"""
Configuration module for Active Inference + Diffusion
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import torch


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process"""
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # Options: "cosine", "linear"
    prediction_type: str = "score"
    use_continuous_time: bool = True
    time_annealing_start: float = 1.0
    time_annealing_end: float = 0.1
    annealing_steps: int = 100000
    gradient_clip_val: float = 0.1
    
@dataclass
class BeliefDynamicsConfig:
    """Configuration for belief dynamics"""
    use_belief_dynamics: bool = True
    belief_dim: int = 50
    diffusion_coefficient: float = 0.1
    learning_rate: float = 0.1
    dt: float = 0.01
    min_variance: float = 1e-6
    max_variance: float = 10.0
    use_full_covariance: bool = False
    noise_scale: float = 0.01
    
@dataclass
class ActiveInferenceConfig:
    """Enhanced configuration for diffusion active inference"""
    # Environment
    env_name: str = "HalfCheetah-v4"
    observation_dim: int = 17  # Add this field
    action_dim: int = 6
    
    # Active inference parameters  
    precision_init: float = 1.0
    expected_free_energy_horizon: int = 5
    efe_horizon: int = 5  # Alias for compatibility
    epistemic_weight: float = 0.1
    extrinsic_weight: float = 1.0
    pragmatic_weight: float = 1.0  # 
    consistency_weight: float = 0.1  # latent policy coherence
    discount_factor: float = 0.99
    contrastive_weight: float = 0.5  # for latent policy coherence
    # Diffusion integration
    kl_weight: float = 0.1  # kl regularization for diffusion
    diffusion_weight: float = 1.0  # score matching weight
    reward_weight:float = 0.5  # reward scaling
    # Model architecture
    hidden_dim: int = 512
    latent_dim: int = 128
    num_layers: int = 3
    pixel_observation: bool = False  # Use pixel observations
    # Training
    batch_size: int = 256
    learning_rate: float = 3e-4
    gradient_clip: float = 1.0

    # Reward-oriented Active Inference parameters
    preference_temperature: float = 1.0  # τ in P(o) ∝ exp(r(o)/τ)
    preference_learning_rate: float = 0.01  # For adaptive temperature
    min_preference_temperature: float = 0.1  # Lower bound for exploration
    max_preference_temperature: float = 10.0  # Upper bound for exploration
    temperature_decay: float = 0.995  # Exponential decay per episode
    use_reward_preferences: bool = True  # Enable reward-oriented EFE
    
    # Preference shaping parameters
    baseline_reward: float = 0.0  # Baseline for reward centering
    preference_momentum: float = 0.9  # EMA for reward statistics    
    # Nested configs
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    belief_dynamics: BeliefDynamicsConfig = field(default_factory=BeliefDynamicsConfig)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class PixelObservationConfig:
    """Configuration for pixel observations"""
    image_shape: Tuple[int, int, int] = (3, 84, 84)
    frame_stack: int = 3
    encoder_type: str = "multiview"  # drqv2, impala, attention
    encoder_feature_dim: int = 50
    augmentation: bool = True
    random_shift_pad: int = 4
    

@dataclass
class TrainingConfig:
    """Training configuration"""
    # General
    total_timesteps: int = 1_000_000
    eval_frequency: int = 10_000
    save_frequency: int = 50_000
    log_frequency: int = 1_000
    
    # Exploration
    exploration_noise: float = 0.1
    exploration_decay: float = 0.999
    min_exploration: float = 0.01
    
    # Buffer
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    train_frequency: int = 2
    gradient_steps: int = 2
    
    # Evaluation
    num_eval_episodes: int = 10
    
    # Logging
    use_wandb: bool = True
    project_name: str = "active-inference-diffusion"
    experiment_name: Optional[str] = None