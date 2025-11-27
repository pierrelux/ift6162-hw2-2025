# Flash Calciner: Physics Model and Neural Surrogate

This document describes the physics-based model of the flash clay calciner and the neural surrogate that approximates its dynamics for fast model predictive control.

## Flash Calciner Physics

A flash calciner is a vertical reactor in which fine clay particles are heated by a counter-current stream of hot gas. The primary chemical transformation is the dehydroxylation of kaolinite:

$$
\text{Al}_2\text{Si}_2\text{O}_5(\text{OH})_4 \longrightarrow \text{Al}_2\text{Si}_2\text{O}_7 + 2\text{H}_2\text{O}
$$

Kaolinite loses its hydroxyl groups and becomes metakaolin, releasing water vapor into the gas phase. This reaction is endothermic and proceeds at temperatures above approximately 500°C, with reaction rate increasing sharply with temperature according to Arrhenius kinetics.

### Spatial Domain and State Variables

The reactor has length $L = 10$ m. We discretize the spatial domain into $N_z = 20$ cells and track five chemical species at each cell:

| Index | Species | Phase | Description |
|-------|---------|-------|-------------|
| 0 | Kaolinite | Solid | Reactant clay mineral |
| 1 | Quartz | Solid | Inert silica |
| 2 | Metakaolin | Solid | Product of dehydroxylation |
| 3 | $\text{N}_2$ | Gas | Inert carrier gas |
| 4 | $\text{H}_2O$ | Gas | Product of dehydroxylation |

At each spatial cell $j \in \{1, \ldots, N_z\}$, we track:

- Concentrations $c_i^{(j)}$ for each species $i \in \{0, \ldots, 4\}$ in mol/m$^3$
- Solid temperature $T_s^{(j)}$ in Kelvin
- Gas temperature $T_g^{(j)}$ in Kelvin

The full state vector $x \in \mathbb{R}^{140}$ concatenates all concentrations and temperatures:

$$
x = \bigl( c_0^{(1)}, \ldots, c_0^{(N_z)}, \ldots, c_4^{(1)}, \ldots, c_4^{(N_z)}, T_s^{(1)}, \ldots, T_s^{(N_z)}, T_g^{(1)}, \ldots, T_g^{(N_z)} \bigr)
$$

### Reaction Kinetics

The dehydroxylation reaction follows third-order Arrhenius kinetics with respect to kaolinite concentration. At temperature $T$ and kaolinite concentration $c_{\text{kao}}$, the reaction rate is

$$
r(T, c_{\text{kao}}) = A \exp\left( -\frac{E_a}{RT} \right) c_{\text{kao}}^3
$$

where $A = 2.9 \times 10^{15}$ $s^{-1}$ is the pre-exponential factor, $E_a = 202$ kJ/mol is the activation energy, and $R = 8.314$ J/(mol·K) is the universal gas constant. The third-order dependence captures the cooperative nature of dehydroxylation across adjacent hydroxyl groups in the crystal lattice.

### Mass Balance

Species concentrations evolve according to convection and reaction. For a solid species $i$ with velocity $v_s$:

$$
\frac{\partial c_i}{\partial t} = -v_s \frac{\partial c_i}{\partial z} + \nu_i \, r(T_s, c_{\text{kao}})
$$

where $\nu_i$ is the stoichiometric coefficient: $\nu_{\text{kao}} = -1$, $\nu_{\text{metakaolin}} = +1$, and $\nu_{\text{H}_2\text{O}} = +2$ (with quartz and N$_2$ inert).

Gas species follow a similar equation with gas velocity $v_g$. The spatial derivatives are discretized using an upwind finite volume scheme on the $N_z$ cells.

### Energy Balance

Heat transfer between the solid and gas phases drives the temperature evolution. The solid temperature at cell $j$ satisfies

$$
\frac{\partial T_s^{(j)}}{\partial t} = -\frac{v_s C_{p,s}}{C_{p,s}} \frac{T_s^{(j)} - T_s^{(j-1)}}{\Delta z} + \frac{h_{sg}(T_g^{(j)} - T_s^{(j)})}{C_{p,s}} - \frac{r \, \Delta H_{\text{rxn}}}{C_{p,s}}
$$

where $C_{p,s}$ is the volumetric heat capacity of the solid phase, $h_{sg}$ is the solid-gas heat transfer coefficient, and $\Delta H_{\text{rxn}}$ is the reaction enthalpy. The first term represents convective transport, the second term is interphase heat transfer, and the third term accounts for the endothermic reaction.

The gas temperature follows a corresponding equation with opposite sign on the heat transfer term, since energy leaving the gas enters the solid.

### Heat Transfer Coefficient

The volumetric heat transfer coefficient between phases depends on the solid volume fraction $\phi_s$ and particle radius $r_b$:

$$
h_{sg} = k_{sg} \cdot \frac{3 \phi_s}{r_b}
$$

where $k_{sg} = 50$ W/(m$^2$·K) is the surface heat transfer coefficient and $r_b = 50$ μm is the particle radius. The factor $3\phi_s / r_b$ represents the specific surface area per unit volume for spherical particles.

### Thermophysical Properties

Heat capacities are computed from NIST polynomial correlations. For solid species, the molar heat capacity takes the form

$$
C_p(T) = k_1 + k_2 T + k_3 T^2 + \frac{k_4}{T} + \frac{k_5}{T^2} + \frac{k_6}{\sqrt{T}}
$$

with coefficients from the NIST Chemistry WebBook. Gas species use the Shomate equation with temperature scaled by 1000 K.

### Control Inputs

The control vector $u \in \mathbb{R}^2$ consists of:

- Gas inlet temperature $T_{g,\text{in}} \in [900, 1350]$ K
- Solid inlet temperature $T_{s,\text{in}} \in [550, 800]$ K

The gas inlet temperature is the primary control variable. Higher temperatures accelerate the reaction but consume more energy in the hot gas generator.

### Boundary Conditions

At the reactor inlet ($z = 0$), concentrations and temperatures are specified by the inlet conditions. The solid enters cold and encounters the hot gas flowing in the opposite direction. At the outlet ($z = L$), we impose zero-gradient (outflow) conditions.

## Neural Surrogate

Simulating the physics model requires integrating a 140-dimensional system of stiff ODEs. A single simulation step of duration $\Delta t = 0.1$ s takes approximately 25 ms on CPU using adaptive Runge-Kutta methods. For model predictive control with horizon $H = 12$ and sample count $K = 96$, each MPC solve would require $H \times K = 1152$ simulation calls, totaling nearly 30 seconds per control step.

A neural surrogate replaces the expensive ODE integration with a single forward pass through a neural network. The surrogate learns the discrete-time dynamics

$$
x_{k+1} = f_\theta(x_k, u_k)
$$

from data collected by running the physics simulator with random initial conditions and control sequences.

### Residual Learning

The state changes only slightly over a single time step $\Delta t = 0.1$ s. Learning the full next state $x_{k+1}$ directly would require the network to accurately reproduce the large baseline values of concentrations and temperatures. Instead, we train the network to predict the state residual:

$$
\Delta x = x_{k+1} - x_k = g_\theta(x_k, u_k)
$$

The predicted next state is then $\hat{x}_{k+1} = x_k + g_\theta(x_k, u_k)$. This formulation improves numerical conditioning and accelerates training convergence.

### Input Normalization

The state vector contains quantities with different scales: concentrations range from 0.01 to 10 $\text{mol/m}^3$, while temperatures range from 600 to 1300 K. We normalize inputs by subtracting the empirical mean and dividing by the empirical standard deviation computed over the training data:

$$
\tilde{x} = \frac{x - \mu_x}{\sigma_x}, \qquad \tilde{u} = \frac{u - \mu_u}{\sigma_u}
$$

The network operates on normalized inputs $(\tilde{x}, \tilde{u})$ and outputs a normalized residual, which is then denormalized to obtain the predicted state increment.

### Spatially-Aware Architecture

The state vector has spatial structure: concentrations and temperatures vary along the reactor length. A standard multilayer perceptron ignores this structure. We instead use one-dimensional convolutions that process the spatial dimension explicitly.

The network takes as input:

- Concentrations reshaped to $(5, N_z)$: five species across $N_z$ cells
- Temperatures reshaped to $(2, N_z)$: solid and gas temperatures across cells
- Control inputs broadcast to $(2, N_z)$: replicated across spatial cells

For concentration prediction, the input tensor of shape $(9, N_z)$ passes through three convolutional layers with kernel size 3, GELU activations, and 10% dropout for regularization. Temperature prediction uses a parallel pathway with the same architecture. The outputs are flattened and concatenated to form the predicted residual.

This architecture has 69,000 parameters and respects the spatial locality of the PDE: neighboring cells interact through convolution kernels, matching the finite-volume discretization of the physics model.

### Training Data Generation

We generate training data by simulating 200 trajectories of length 50, yielding 10,000 state transitions. Each trajectory starts from one of three initial condition modes:

| Mode | Probability | Description |
|------|-------------|-------------|
| Cold start | 50% | Uniform low temperature (550-700 K), unreacted concentrations |
| Warm front | 25% | Partial reaction front at random position along reactor |
| Steady-state | 25% | Near-equilibrium profiles with spatial gradients |

This diversity is critical: training only on near-steady-state data yields a surrogate that fails catastrophically on cold-start transients (1468% error). Including cold starts and reaction fronts reduces error to 18.8% on challenging 80-step rollouts with time-varying controls.

At each trajectory step, we apply a slowly varying random control sequence with occasional jumps, providing coverage of the operating region.

### Training Procedure

The network is trained for 100 epochs using the AdamW optimizer with learning rate $10^{-3}$ and weight decay $10^{-5}$. The loss function is mean squared error between predicted and true next states in normalized coordinates:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \| f_\theta(\tilde{x}_i, \tilde{u}_i) - \tilde{x}_{i+1} \|^2
$$

A cosine annealing schedule decays the learning rate to zero over the training run. With 10,000 training samples and 69,000 parameters, the sample-to-parameter ratio is approximately 1:7, providing adequate coverage to avoid severe overfitting.

### Accuracy and Speedup

On challenging evaluations with cold-start initial conditions and time-varying controls, the surrogate achieves approximately 19% mean relative error over 80-step rollouts. On near-steady-state conditions, error drops to under 1%. The error is concentrated at later timesteps and near the reactor outlet, where autoregressive prediction errors accumulate.

This accuracy is sufficient for MPC: the controller replans at each time step, and the physics simulator provides ground-truth feedback for closed-loop correction. The trained surrogate evaluates a 20-step rollout in 0.4 ms, compared to 25 ms per step for the physics simulator, representing a 60× speedup.
