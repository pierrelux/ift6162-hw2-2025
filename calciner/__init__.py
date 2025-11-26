"""
Flash Calciner - Neural Surrogate MPC

A package for energy-optimal control of flash clay calciners using:
- Neural surrogate models for fast dynamics prediction
- Model Predictive Control (MPPI, gradient-based)
- Reinforcement Learning baselines (TD3, PPO)

Modules:
    physics: PDE-based flash calciner model
    surrogate: Neural network surrogate + training
    mpc: MPC controllers (MPPI, gradient, classical)
    baselines: Environment and constant-temp baseline
    rl: Reinforcement learning algorithms (TD3, PPO)
"""

from .physics import SimplifiedFlashCalciner, N_SPECIES, L
from .surrogate import (
    CalcinerSimulator,
    SurrogateModel,
    SpatiallyAwareDynamics,
    MLPDynamics,
    TransitionDataset,
    generate_training_data,
    train_surrogate,
)
from .mpc import (
    SurrogateMPPI,
    SurrogateMPC,
    CalcinerDynamics,
    ClassicalMPC,
)
from .baselines import (
    CalcinerEnv,
    ConstantTemperatureController,
    evaluate_baseline,
)

__version__ = "0.1.0"

