# Assignment: Deep Reinforcement Learning for Flash Calciner Control

## Overview

In this assignment, you will implement deep reinforcement learning algorithms to control a flash clay calciner—an industrial reactor that converts kaolinite clay into metakaolin through thermal dehydroxylation. The control objective is to achieve high conversion while minimizing energy consumption.

The assignment has two parts:

| Part | Problem | State Dim | Algorithms |
|------|---------|-----------|------------|
| **Part 1** | Simplified 1D dynamics | 3 | REINFORCE, PPO, TD3 |
| **Part 2** | Full PDE-based dynamics | 140 | Scale up your best algorithm |

## Background

Read `docs/model.md` for a complete description of the flash calciner physics. The key points are:

- **Reaction**: Kaolinite (clay) → Metakaolin + Water vapor (endothermic)
- **Control input**: Gas inlet temperature $T_{g,in} \in [900, 1300]$ K
- **Objective**: Maximize conversion $\alpha$ while minimizing heater power
- **Constraint**: Maintain $\alpha \geq \alpha_{min}$ (time-varying target)

---

## Part 1: Simplified Control Problem (50 points)

### Environment

#### Simplified Dynamics Model

The full flash calciner involves coupled PDEs for mass and energy transport across 20 spatial cells (see `docs/model.md`). For Part 1, we abstract this to a scalar system that captures the essential control challenge.

The **outlet conversion** $\alpha \in [0,1]$ (fraction of kaolinite converted to metakaolin) evolves as a first-order lag approaching a temperature-dependent equilibrium:

$$\alpha_{k+1} = e^{-\Delta t/\tau} \alpha_k + (1 - e^{-\Delta t/\tau}) \, \alpha_{ss}(u_k)$$

where:
- $\tau = 2$ s is the thermal time constant (how fast the reactor responds to temperature changes)
- $\Delta t = 0.5$ s is the control period
- $\alpha_{ss}(u)$ is the **steady-state conversion**: the equilibrium $\alpha$ would reach if gas inlet temperature were held constant at $u$ indefinitely

In other words, $\alpha_{ss}(u) = \lim_{t \to \infty} \alpha(t)$ when applying constant control $u$. The discrete dynamics exponentially relax toward this equilibrium with rate $1/\tau$.

The steady-state conversion follows a **sigmoidal Arrhenius-like relationship**:

$$\alpha_{ss}(T) = \frac{0.999}{1 + \exp(-0.025 \cdot (T - 1000))}$$

This captures two key physics phenomena:
1. **Arrhenius kinetics**: Reaction rate (and thus conversion) increases exponentially with temperature
2. **Saturation**: Conversion cannot exceed 100%, so it approaches an asymptote

Numerically: $\alpha_{ss}(900\text{K}) \approx 50\%$, $\alpha_{ss}(1000\text{K}) \approx 73\%$, $\alpha_{ss}(1261\text{K}) \approx 99.8\%$. The inflection point at $T = 1000$ K corresponds to the temperature where the reaction becomes kinetically favorable.

#### Why This Simplification?

The first-order model is a **linear approximation around the dominant eigenvalue** of the linearized PDE system. It preserves:
1. The fundamental tradeoff: higher $T$ → faster conversion but more energy
2. Temporal dynamics: the system doesn't respond instantaneously
3. The constraint satisfaction problem: track a time-varying $\alpha_{min}(t)$

What it ignores: spatial profiles, species concentrations, solid/gas temperature differences. You'll tackle those in Part 2.

#### Environment API

```python
from calciner import CalcinerEnv

env = CalcinerEnv(episode_length=40, dt=0.5)
obs = env.reset()  # Returns [alpha, alpha_min, t/T]
obs, reward, done, info = env.step(action)  # action = T_g_in in [900, 1300]
```

**State space** $\mathcal{S} \subset \mathbb{R}^3$:
| Component | Description | Range |
|-----------|-------------|-------|
| $\alpha$ | Current outlet conversion | $[0, 1]$ |
| $\alpha_{min}$ | Target minimum (constraint) | $[0.90, 0.99]$ |
| $t/T$ | Normalized episode time | $[0, 1]$ |

**Action space** $\mathcal{A} \subset \mathbb{R}$:
- $u = T_{g,in}$: Gas inlet temperature in $[900, 1300]$ K

**Reward function**:
$$r(s, a) = -\underbrace{c \cdot (T_{g,in} - 300)}_{\text{energy cost}} - \underbrace{10 \cdot \max(0, \alpha_{min} - \alpha)^2}_{\text{constraint violation}}$$

The energy cost is linear in temperature (heating gas from ambient). The constraint penalty is a soft barrier—you can violate $\alpha_{min}$ but it's expensive. This is a **constrained MDP** relaxed via penalty methods.

### Tasks

1. **REINFORCE** (15 points)
   - Implement the REINFORCE algorithm with baseline
   - Use a linear or small neural network policy
   - Train for 200+ episodes and plot learning curves

2. **PPO** (20 points)
   - Implement Proximal Policy Optimization
   - Include: clipped objective, value function, GAE (optional)
   - Compare sample efficiency to REINFORCE

3. **TD3** (15 points)
   - Implement Twin Delayed DDPG
   - Include: twin critics, delayed policy updates, target smoothing
   - Use a replay buffer

### Evaluation

Compare your algorithms against the constant-temperature baseline:

```python
from calciner import ConstantTemperatureController, evaluate_baseline

baseline = ConstantTemperatureController(T_g_in=1261.15)
results = evaluate_baseline(env, baseline)
print(f"Baseline: {results}")
```

Report:
- Learning curves (return vs episode)
- Final policy performance (energy, constraint violations)
- Policy visualization (what temperature does it choose vs. state?)

---

## Part 2: Full 140-Dimensional Problem (50 points)

### The Challenge

The simplified model ignores spatial dynamics. The real calciner is a 10m reactor where temperature and concentration profiles evolve along the length. The full state has 140 dimensions:
- 5 species × 20 spatial cells = 100 concentration values
- 2 temperatures × 20 cells = 40 temperature values

Simulating this PDE-based model is expensive (~25ms per step), making RL training impractical. Instead, you will use a **neural surrogate** that approximates the dynamics in ~0.02ms.

### Loading the Surrogate

```python
import torch
from calciner import SpatiallyAwareDynamics, SurrogateModel, CalcinerSimulator
import numpy as np

# Load trained surrogate
checkpoint = torch.load('models/surrogate_model.pt', weights_only=False)
model = SpatiallyAwareDynamics(N_z=checkpoint['N_z'])
model.load_state_dict(checkpoint['model_state_dict'])

norm_params = {k: np.array(v) for k, v in checkpoint['norm_params'].items()}
surrogate = SurrogateModel(model, norm_params)

# Also have access to the true physics simulator for validation
simulator = CalcinerSimulator(N_z=20, dt=0.1)
```

### Creating an Environment

You need to create a Gym-like environment wrapper around the surrogate. Key considerations:

1. **State representation**: The 140D state may need preprocessing
   - Flatten vs. keep spatial structure?
   - Normalize concentrations and temperatures differently?
   
2. **Reward design**: 
   - Conversion: computed from kaolinite concentration at outlet
   - Energy: function of $T_{g,in}$
   - How to balance the two?

3. **Episode termination**:
   - Fixed horizon?
   - Early termination on constraint violation?

### Tasks

1. **Environment Design** (10 points)
   - Create a Gym-compatible environment using the surrogate
   - Document your state/action/reward design choices

2. **Algorithm Scaling** (25 points)
   - Adapt your best algorithm from Part 1 to the 140D problem
   - You may need:
     - Larger neural networks
     - Different hyperparameters
     - State preprocessing/normalization

3. **Evaluation** (15 points)
   - Validate learned policy on the **true physics simulator**
   - Compare to constant-temperature baseline
   - Visualize: state profiles, control trajectory, conversion over time

### Hints

- The surrogate is differentiable—you could compute policy gradients through the dynamics
- The 140D state has structure: use it (e.g., 1D convolutions like the surrogate itself)
- Start with a simple policy (e.g., just output $T_{g,in}$ as function of outlet conversion)

---

## Deliverables

1. **Code**: Your implementations in a clean, documented format
2. **Report** (max 6 pages): 
   - Algorithm descriptions
   - Experimental setup
   - Results with plots
   - Discussion of what worked/didn't work

3. **Trained models**: Save your best policy checkpoints

---

## Getting Started

```bash
# Install dependencies
pip install numpy torch matplotlib scipy

# Test the environments
python -c "from calciner import CalcinerEnv; env = CalcinerEnv(); print(env.reset())"

# Check the surrogate
python -c "import torch; print(torch.load('models/surrogate_model.pt', weights_only=False).keys())"
```

## Grading

| Component | Points |
|-----------|--------|
| Part 1: REINFORCE | 15 |
| Part 1: PPO | 20 |
| Part 1: TD3 | 15 |
| Part 2: Environment | 10 |
| Part 2: Algorithm | 25 |
| Part 2: Evaluation | 15 |
| **Total** | **100** |

Partial credit is given. A working but suboptimal implementation is better than no implementation.

---

## References

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (TD3, 2018)
- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (REINFORCE, 1992)
- `docs/model.md` for flash calciner physics

