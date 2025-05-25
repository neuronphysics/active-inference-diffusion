# Active Inference + Diffusion for Continuous Control

A framework combining Active Inference principles with Diffusion Models for continuous state-action spaces in reinforcement learning.

## Overview

This package implements Active Inference with the following components:
- Variational Free Energy minimization
- Expected Free Energy for action selection  
- Fokker-Planck belief dynamics
- Score-based diffusion models
- Support for both state and pixel observations

## Installation

```bash
# Clone the repository
git clone https://github.com/neuronphysics/active-inference-diffusion.git
cd active-inference-diffusion

# Install in development mode
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

## Quick Start

### State-Based Control

```python
from active_inference_diffusion.agents import StateBasedAgent
from active_inference_diffusion.configs import ActiveInferenceConfig, TrainingConfig
import gymnasium as gym

# Create environment
env = gym.make('HalfCheetah-v4')

# Configure
config = ActiveInferenceConfig(env_name='HalfCheetah-v4')
training_config = TrainingConfig()

# Create and train agent
agent = StateBasedAgent(env, config, training_config)
```

### Pixel-Based Control

```python
from active_inference_diffusion.agents import PixelBasedAgent
from active_inference_diffusion.envs import make_pixel_mujoco
from active_inference_diffusion.configs import PixelObservationConfig

# Create pixel environment
env = make_pixel_mujoco('HalfCheetah-v4', width=84, height=84, frame_stack=3)

# Additional pixel configuration
pixel_config = PixelObservationConfig()

# Create agent
agent = PixelBasedAgent(env, config, training_config, pixel_config)
```

## Training

### Command Line

```bash
# Train with state observations
python examples/train_state_mujoco.py --env HalfCheetah-v4

# Train with pixel observations  
python examples/train_pixel_mujoco.py --env HalfCheetah-v4

# Resume from checkpoint
python examples/train_state_mujoco.py --env HalfCheetah-v4 --resume

# Use custom config
python examples/train_pixel_mujoco.py --env Hopper-v4 --config examples/configs/hopper_pixel.yaml
```

### Configuration Files

Example configuration structure:

```yaml
active_inference:
  env_name: "HalfCheetah-v4"
  expected_free_energy_horizon: 5
  epistemic_weight: 0.1
  extrinsic_weight: 1.0
  
training:
  total_timesteps: 1_000_000
  batch_size: 256
  learning_rate: 3e-4
```

## Key Components

### Active Inference
- **Free Energy**: F = Complexity - Accuracy
- **Expected Free Energy**: G = Epistemic Value - Extrinsic Value  
- **Belief Dynamics**: Continuous updates using Fokker-Planck equation

### Diffusion Models
- Score-based generative modeling
- Stable training with score matching
- Flexible noise schedules (linear, cosine, sigmoid)

### Visual Learning
- DrQ-v2 encoder for pixel observations
- Contrastive representation learning
- Data augmentation for sample efficiency

## Project Structure

```
active-inference-diffusion/
├── active_inference_diffusion/
│   ├── core/               # Core active inference implementation
│   ├── models/             # Neural network models
│   ├── agents/             # Agent implementations  
│   ├── encoders/           # State and visual encoders
│   ├── envs/               # Environment wrappers
│   └── utils/              # Utilities and helpers
│   
├── examples/               # Training scripts and configs
|    └── configs/           # Configuration classes
├── setup.py               # Package setup
└── requirements.txt       # Dependencies
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- Gymnasium >= 0.28.0
- MuJoCo >= 2.3.0
- NumPy, matplotlib, tqdm, wandb

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{active_inference_diffusion,
  title={Active Inference + Diffusion for Continuous Control},
  author={Zahra Sheikhbahaee},
  year={2025},
  url={https://github.com/neuronphysics/active-inference-diffusion}
}
```