# Assignment: Deep RL for Flash Calciner Control

**Due: December 10, 2025**

Industrial reactors are expensive to run. A flash calciner heats clay at 900-1300 K to produce metakaolin (used in high-performance concrete). Your job: learn a control policy that achieves target conversion $\alpha \geq \alpha_{min}$ while minimizing energy.

This is a constrained control problem with continuous actions. You'll implement three RL algorithms on a simplified 3D problem, then scale your best approach to the full 140D PDE-based system using a neural surrogate.

| Part | State | Task | Points |
|------|-------|------|--------|
| **1** | 3D (scalar dynamics) | Implement REINFORCE, PPO, TD3 | 50 |
| **2** | 140D (PDE surrogate) | Scale up your best algorithm | 50 |

**Setup**: Run `python scripts/demo.py` to test environments. See `docs/model.md` for physics background.

**Physics model**: Based on Cantisani et al., "Dynamic modeling and simulation of a flash clay calciner" (2018). The physics implementation is reproduced from the paper at https://github.com/pierrelux/flash-clay-calciner this assignment adds control on top of that dynamics model.

## Logistics

You may work in teams of up to 3 people. All team members must understand the solution and code. We may conduct individual interview checkups where you'll be asked to explain your implementation choices, algorithm details, and results. If you worked in a team and cannot explain your own code, you will receive a reduced grade.

## Part 1: Simplified Problem (50 points)

### The Dynamics

Outlet conversion $\alpha \in [0,1]$ follows first-order lag dynamics:

$$\alpha_{k+1} = e^{-\Delta t/\tau} \alpha_k + (1 - e^{-\Delta t/\tau}) \, \alpha_{ss}(u_k)$$

where $\tau = 2$ s, $\Delta t = 0.5$ s, and steady-state conversion $\alpha_{ss}(T) = 0.999/(1 + \exp(-0.025(T - 1000)))$ captures Arrhenius kinetics. At 900K you get ~50% conversion; at 1200K you approach 100%.

**Environment**: State is $[\alpha, \alpha_{min}, t/T] \in \mathbb{R}^3$. Action is $T_{g,in} \in [900, 1300]$ K. Reward is $-\text{energy} - 10 \cdot \max(0, \alpha_{min} - \alpha)^2$. Import with `from calciner import CalcinerEnv`.

### The Algorithms

**REINFORCE** (15 pts): Policy gradient with baseline. Use Monte Carlo returns. Train 200+ episodes. This should work but will be sample-inefficient.

**PPO** (20 pts): Clipped surrogate objective + value network. More stable than vanilla policy gradient. Should converge faster than REINFORCE.

**TD3** (15 pts): Twin critics, delayed updates, target smoothing. Use replay buffer. Most sample-efficient but requires tuning exploration noise.

### Evaluation

Compare against `ConstantTemperatureController(T_g_in=1261)` which achieves 99.8% conversion but uses maximum energy. Can your policies match 95% conversion with 50% less energy?

Plot learning curves. Does the policy learn to lower temperature when $\alpha$ exceeds $\alpha_{min}$, or does it just blast heat?

---

## Part 2: Full 140D Problem (50 points)

### The Challenge

The full state tracks 5 chemical species + 2 temperature profiles across 20 spatial cells (140D). The physics simulator is too slow (~25 ms/step) for RL. We provide a neural surrogate that runs 60× faster with 19% error on hard rollouts, <1% near steady-state. Use `SurrogateCalcinerEnv` which wraps the surrogate.

### Algorithm (25 pts)

Adapt your best Part 1 algorithm to 140D. You'll need a neural network policy—linear won't work. Architecture choices matter:

- **Fully-connected**: Simple, but ignores spatial structure. Try 2-3 layers with 128-256 units.
- **1D convolutions**: Exploits spatial locality (like the surrogate does). Harder to implement but potentially more sample-efficient.
- **Hybrid**: Extract outlet conversion + temperature gradient, feed into MLP.

Start simple. A policy that only looks at state[19] (outlet concentration) can work as a baseline. Then add more state features.

### Evaluation (15 pts)

Validate on `CalcinerSimulator` (the true physics), not just the surrogate. The policy must generalize beyond the neural model's approximation errors.

Visualize the closed-loop spatial profiles. Does your policy create smooth temperature gradients along the reactor, or does it cause sharp fronts that could damage equipment? This matters in real process control.

Report final energy consumption and constraint violations vs. the constant-temperature baseline.

---

## What to Submit

Your code and a markdown file summarizing:
- Which algorithms you implemented and how
- Hyperparameters and design choices (network architecture, learning rates, etc.)
- Learning curves comparing the three algorithms
- Final performance metrics (energy, violations) vs baseline
- For Part 2: does the 140D policy exploit spatial structure? Show spatial profile visualizations
- What worked, what didn't, what you'd try next

Save your trained model checkpoints so we can reproduce your best policies.

## Grading

REINFORCE (15), PPO (20), TD3 (15), Part 2 algorithm (25), Part 2 evaluation (15). Partial credit given.

## Tips

- Start with a simple policy architecture. Even a 2-layer MLP can work for Part 1.
- For Part 2, the state is already normalized by the environment. Don't re-normalize.
- The baseline achieves 99.8% conversion but uses maximum energy. Can you match 95% conversion with 50% less energy?
- Visualizations matter: plot temperature vs time, not just scalar rewards.
- The surrogate has 19% error on hard rollouts but <1% on steady-state. Your policy will naturally stay near easier-to-predict regions.
- Test on the true physics simulator! The surrogate is just for training speed.