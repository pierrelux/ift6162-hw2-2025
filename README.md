# Flash Clay Calciner Model

Dynamic modeling and simulation of a flash clay calciner based on Cantisani et al.

## Model

The implementation includes:
- Reaction kinetics (third-order Arrhenius for kaolinite dehydroxylation)
- Thermophysical properties (NIST Shomate for gas, polynomial for solid)
- Mass and energy balances
- Spatial discretization (finite volume method)

## Usage

```bash
pip install -r requirements.txt
python flash_calciner.py
```

Generates:
- `figure3_states_3d.png`: Dynamic states in time and space
- `figure4_steady_state.png`: Steady-state reaction rate and temperature profiles
- `figure5_concentrations.png`: Concentration profiles

## Reference

Cantisani, N., Svensen, J. L., Hansen, O. F., & Jørgensen, J. B. (2024). Dynamic modeling and simulation of a flash clay calciner.

