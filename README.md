# Flash Clay Calciner - Neural Surrogate MPC

Economic Model Predictive Control for flash clay calciner using a learned neural surrogate.

## Overview

This project implements:
1. **Physics-based PDE model** of a flash clay calciner (140-dimensional state)
2. **Neural surrogate** that learns discrete-time dynamics (61× faster than physics)
3. **MPPI controller** for energy-optimal control with conversion constraints
4. **RL baselines** (TD3, PPO) for comparison

## Key Results

| Metric | Value |
|--------|-------|
| Surrogate speedup | **61×** (25ms vs 1.5s per rollout) |
| Energy savings | **72%** vs constant-temperature baseline |
| Conversion target | ✓ Achieved (96.8% vs 95% target) |
| MPC solve time | 1.5s/step (CPU) |

## Project Structure

```
├── calciner/                      # Main package
│   ├── __init__.py
│   ├── physics.py                 # PDE-based flash calciner model
│   ├── surrogate.py               # Neural surrogate + training
│   ├── mpc.py                     # MPC controllers (MPPI, gradient, classical)
│   ├── baselines.py               # Env for RL, constant-temp baseline
│   └── rl/                        # RL algorithms
│       ├── __init__.py
│       ├── td3.py                 # TD3 (Twin Delayed DDPG)
│       └── ppo.py                 # PPO (Proximal Policy Optimization)
│
├── scripts/                       # Entry point scripts
│   ├── train.py                   # Train neural surrogate
│   ├── evaluate_mpc.py            # Run MPC evaluation
│   └── evaluate_rl.py             # Train and evaluate RL baselines
│
├── models/                        # Trained model weights
│   └── surrogate_model.pt
│
├── figures/                       # Output figures
│
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train surrogate model
python scripts/train.py

# Evaluate MPC controller
python scripts/evaluate_mpc.py

# Train and evaluate RL baselines
python scripts/evaluate_rl.py
```

## Method

### 1. Neural Surrogate

Learns discrete-time dynamics of the 140D PDE system:
```
x_{t+1} = f_θ(x_t, u_t)
```

- **Architecture**: Spatially-aware 1D convolutions (261K params)
- **Training**: Residual learning on 900 physics simulation transitions
- **Accuracy**: ~10% relative error (sufficient for MPC)

### 2. MPPI Controller

Model Predictive Path Integral control:
- Samples 96 random control sequences
- Rolls out trajectories using surrogate (in parallel)
- Weights by cost, returns weighted average

**Cost function**: Energy + soft conversion constraint + terminal temperature

### 3. RL Baselines

For comparison, we also implement:
- **TD3**: Off-policy actor-critic with twin critics
- **PPO**: On-policy with clipped objective

## Physics Model

Based on Cantisani et al. "Dynamic modeling and simulation of a flash clay calciner":

- **Reaction**: Kaolinite → Metakaolin + 2H₂O (3rd order Arrhenius)
- **State**: 5 species × 20 cells + 2 temperatures × 20 cells = 140D
- **Discretization**: Finite volume with upwind convection

## Reference

Cantisani, N., Svensen, J. L., Hansen, O. F., & Jørgensen, J. B. (2024). 
Dynamic modeling and simulation of a flash clay calciner.
