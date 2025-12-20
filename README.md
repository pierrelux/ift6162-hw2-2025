# Flash Calciner Control

Deep reinforcement learning for industrial process control.

![Physics vs Surrogate Comparison](figures/surrogate_comparison.png)

*Neural surrogate learns to approximate the 140D physics simulator with 19% mean relative error, enabling 60× faster rollouts for RL training.*

## Assignment

See [ASSIGNMENT.md](ASSIGNMENT.md) for the full problem statement.

**TL;DR**: Implement REINFORCE, PPO, and TD3 to control a flash clay calciner.

## Quick Start

```bash
# Install dependencies
pip install numpy torch matplotlib scipy

# Test Part 1 environment
python -c "
from calciner import CalcinerEnv
env = CalcinerEnv()
obs = env.reset()
print(f'Initial state: {obs}')
for _ in range(5):
    obs, reward, done, info = env.step(1100.0)
    print(f'  α={info[\"alpha\"]:.2%}, reward={reward:.2f}')
"

# Test surrogate loading
python -c "
import torch
ckpt = torch.load('models/surrogate_model.pt', weights_only=False)
print(f'Surrogate: {ckpt[\"N_z\"]}×140D state, dt={ckpt[\"dt\"]}s')
"
```

## Project Structure

```
├── ASSIGNMENT.md          # ← Start here
├── calciner/
│   ├── physics.py         # Full PDE-based simulator (given)
│   ├── surrogate.py       # Neural surrogate architecture (given)
│   ├── mpc.py             # Simplified dynamics model (given)
│   └── baselines.py       # Part 1 RL environment (given)
├── models/
│   └── surrogate_model.pt # Trained surrogate checkpoint (given)
├── docs/
│   └── model.md           # Physics documentation
└── figures/               # Reference results
```

## What's Provided

| Component | Description |
|-----------|-------------|
| `CalcinerEnv` | Gym-like environment for Part 1 (simplified 1D) |
| `CalcinerSimulator` | Full 140D physics simulator |
| `SurrogateModel` | Trained neural surrogate for fast simulation |
| `ConstantTemperatureController` | Baseline controller for comparison |

## What You Implement

| Algorithm | Part | Points |
|-----------|------|--------|
| REINFORCE | 1 | 15 |
| PPO | 1 | 20 |
| TD3 | 1 | 15 |
| 140D Environment | 2 | 10 |
| Scaled Algorithm | 2 | 25 |
| Evaluation | 2 | 15 |

## Tips

1. **Part 1**: Start simple. A linear policy can work for the 3D state.
2. **Part 2**: The surrogate is differentiable—you can backprop through dynamics.
3. **Debugging**: Visualize your policy's behavior, not just the reward curve.
4. **Baselines**: Always compare to the constant-temperature controller.
