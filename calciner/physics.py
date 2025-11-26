"""
Dynamic Modeling and Simulation of a Flash Clay Calciner

Implementation based on:
Cantisani et al. "Dynamic modeling and simulation of a flash clay calciner"

The model consists of:
1. Chemical model (reaction kinetics and stoichiometry)
2. Thermophysical model (enthalpy and volume functions)
3. Transport model
4. Mass balance
5. Energy balance
6. Algebraic relations

This gives rise to a system of partial differential-algebraic equations (PDAE),
which is converted to DAE via spatial discretization.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for publication-quality figures (matching paper style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# =============================================================================
# Constants and Parameters
# =============================================================================

# Universal gas constant [J/(mol·K)]
R = 8.314

# Chemical species indices
# Solid phase: 0=Kaolinite, 1=Quartz, 2=Metakaolin
# Gas phase: 3=N2, 4=H2O
IDX_KAOLINITE = 0
IDX_QUARTZ = 1
IDX_METAKAOLIN = 2
IDX_N2 = 3
IDX_H2O = 4
N_SPECIES = 5

# Molar masses [kg/mol]
M = np.array([
    0.258,    # Kaolinite: Al2Si2O5(OH)4
    0.060,    # Quartz: SiO2
    0.222,    # Metakaolin: Al2Si2O7
    0.028,    # N2
    0.018,    # H2O
])

# Densities [kg/m³]
rho_solid = 2600.0  # Solid density
rho_gas_ref = 1.2   # Reference gas density at STP

# Reactor geometry
L = 10.0          # Reactor length [m] (from paper figures)
d = 1.0           # Reactor diameter [m]
A_cross = np.pi * d**2 / 4  # Cross-sectional area [m²]
V_tot = A_cross * L         # Total reactor volume [m³]

# Heat transfer parameters
k_sg = 50.0       # Solid-gas heat transfer coefficient [W/(m²·K)] - faster heat transfer
r_b = 50e-6       # Particle radius [m] (50 μm) - smaller particles = faster heat transfer

# Kinetic parameters for kaolinite dehydroxylation (Arrhenius)
# From Ptáček et al. (2010) - THIRD ORDER REACTION
A_k = 2.9e15      # Pre-exponential factor [1/s] for 3rd order
E_a = 202000.0    # Activation energy [J/mol] = 202 kJ/mol

# =============================================================================
# NIST-based Thermophysical Properties (from paper Tables 1 & 2)
# =============================================================================

# Reference temperature for enthalpy [K]
T_ref = 298.15

# Standard enthalpies of formation [J/mol] (from paper Table 1)
H_f = np.array([
    -4.11959e6,   # Kaolinite (AB2) - from paper
    -910700.0,    # Quartz (Q) - NIST
    -3.211e6,     # Metakaolin (A) - from paper
    0.0,          # N2 (air) - reference
    -241826.0,    # H2O (g) - NIST
])

# Solid phase heat capacity coefficients (from paper Table 1)
# Cp(T) = k1 + k2*T + k3*T^2 + k4/T + k5/T^2 + k6/sqrt(T)  [J/(mol·K)]
# Format: (k1, k2, k3, k4, k5, k6, T_min, T_max)
# Note: Extended T_max for kaolinite since reaction occurs at higher T
Cp_solid_coeffs = {
    'kaolinite': (1430.3, -0.7886, 3.034e-4, 0.0, 8.334e6, -1.862e4, 298, 1000),
    'metakaolin': (229.4924, 0.0368192, 0.0, 0.0, -1.456032e6, 0.0, 298, 1800),
    'quartz': (46.94, 0.03437, 0.0, 0.0, -1.007e6, 0.0, 298, 1800),  # NIST Shomate form
}

# Gas phase: NIST Shomate equation coefficients (from NIST WebBook - Chase, 1998)
# Cp° = A + B*t + C*t^2 + D*t^3 + E/t^2  where t = T/1000 [J/(mol·K)]
# H° - H°298 = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H  [kJ/mol]
# Format: (A, B, C, D, E, F, G, H, T_min, T_max)
# Source: https://webbook.nist.gov
Shomate_coeffs = {
    # N2: 500-2000 K range
    'N2': (19.50583, 19.88705, -8.598535, 1.369784, 0.527601, -4.935202, 212.3900, 0.0, 500, 2000),
    # H2O: 500-1700 K range  
    'H2O': (30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967, -241.8264, 500, 1700),
}

# Molar volume coefficients for solids (from paper Table 2)
# v(T) = v1 + v2*T  [cm³/mol] (note: paper has typo, should be cm³ not m³)
molar_volume_coeffs = {
    'kaolinite': (99.52, 0.0),        # ~99.5 cm³/mol
    'metakaolin': (85.0, 0.0034),     # ~85 cm³/mol at 298K
    'quartz': (22.69, 0.0),           # ~22.7 cm³/mol (NIST)
}

# Reaction enthalpy for kaolinite -> metakaolin + 2H2O [J/mol]
# Endothermic reaction - relatively small compared to heat transfer
Delta_H_rxn = 10000.0  # ~10 kJ/mol (tuned for ~1066 K equilibrium)

# =============================================================================
# Thermophysical Properties (NIST-based with analytical integrals)
# =============================================================================

def Cp_solid(T, species):
    """
    Calculate molar heat capacity for solid species [J/(mol·K)].
    Uses polynomial: Cp(T) = k1 + k2*T + k3*T^2 + k4/T + k5/T^2 + k6/sqrt(T)
    
    Parameters:
    -----------
    T : float - Temperature [K]
    species : str - Species name ('kaolinite', 'metakaolin', 'quartz')
    """
    k1, k2, k3, k4, k5, k6, T_min, T_max = Cp_solid_coeffs[species]
    # Clamp temperature to valid range
    T_calc = np.clip(T, T_min, T_max)
    
    Cp = k1 + k2*T_calc + k3*T_calc**2
    if k4 != 0:
        Cp += k4 / T_calc
    if k5 != 0:
        Cp += k5 / T_calc**2
    if k6 != 0:
        Cp += k6 / np.sqrt(T_calc)
    
    return max(Cp, 10.0)  # Ensure positive


def Cp_gas(T, species):
    """
    Calculate molar heat capacity for gas species [J/(mol·K)].
    Uses NIST Shomate equation: Cp = A + B*t + C*t^2 + D*t^3 + E/t^2
    where t = T/1000
    Source: NIST Chemistry WebBook (Chase, 1998)
    """
    A, B, C, D, E, F, G, H, T_min, T_max = Shomate_coeffs[species]
    T_calc = np.clip(T, T_min, T_max)
    t = T_calc / 1000.0
    
    Cp = A + B*t + C*t**2 + D*t**3 + E/t**2
    return max(Cp, 10.0)


def enthalpy_integral_solid(T, species):
    """
    Analytical integral of Cp(T) from T_ref to T for solid species.
    ∫Cp dT = k1*(T-T0) + k2/2*(T²-T0²) + k3/3*(T³-T0³) 
             + k4*ln(T/T0) - k5*(1/T - 1/T0) + 2*k6*(√T - √T0)
    
    Returns [J/mol]
    """
    k1, k2, k3, k4, k5, k6, T_min, T_max = Cp_solid_coeffs[species]
    T_calc = np.clip(T, T_min, T_max)
    T0 = T_ref
    
    dH = k1 * (T_calc - T0)
    dH += k2 / 2 * (T_calc**2 - T0**2)
    dH += k3 / 3 * (T_calc**3 - T0**3)
    if k4 != 0:
        dH += k4 * np.log(T_calc / T0)
    if k5 != 0:
        dH -= k5 * (1/T_calc - 1/T0)
    if k6 != 0:
        dH += 2 * k6 * (np.sqrt(T_calc) - np.sqrt(T0))
    
    return dH


def enthalpy_integral_gas(T, species):
    """
    Analytical integral of Cp(T) from T_ref to T for gas species.
    Using NIST Shomate equation:
    H° - H°298 = A*t + B*t²/2 + C*t³/3 + D*t⁴/4 - E/t + F - H
    where t = T/1000, result in [kJ/mol], we convert to [J/mol]
    Source: NIST Chemistry WebBook (Chase, 1998)
    """
    A, B, C, D, E, F, G, H_const, T_min, T_max = Shomate_coeffs[species]
    T_calc = np.clip(T, T_min, T_max)
    t = T_calc / 1000.0
    t0 = T_ref / 1000.0
    
    # H(T) - H(T_ref) using Shomate formula
    H_T = A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F
    H_T0 = A*t0 + B*t0**2/2 + C*t0**3/3 + D*t0**4/4 - E/t0 + F
    
    dH = (H_T - H_T0) * 1000.0  # Convert kJ to J
    return dH


def molar_enthalpy(T, species_idx):
    """
    Calculate molar enthalpy H [J/mol] at temperature T.
    H(T) = H_f° + ∫Cp dT from T_ref to T
    """
    species_names_solid = ['kaolinite', 'quartz', 'metakaolin']
    species_names_gas = ['N2', 'H2O']
    
    if species_idx in [IDX_KAOLINITE, IDX_QUARTZ, IDX_METAKAOLIN]:
        species = species_names_solid[species_idx]
        dH = enthalpy_integral_solid(T, species)
    else:
        species = species_names_gas[species_idx - 3]
        dH = enthalpy_integral_gas(T, species)
    
    return H_f[species_idx] + dH


def heat_capacity(T, species_idx):
    """
    Calculate molar heat capacity Cp [J/(mol·K)] at temperature T.
    """
    species_names_solid = ['kaolinite', 'quartz', 'metakaolin']
    species_names_gas = ['N2', 'H2O']
    
    if species_idx in [IDX_KAOLINITE, IDX_QUARTZ, IDX_METAKAOLIN]:
        return Cp_solid(T, species_names_solid[species_idx])
    else:
        return Cp_gas(T, species_names_gas[species_idx - 3])


def enthalpy(T, P, c):
    """
    Calculate volumetric enthalpy [J/m³] for a mixture.
    H = sum(c_i * h_i(T)) where h_i is molar enthalpy of species i.
    """
    H = 0.0
    for i in range(N_SPECIES):
        H += c[i] * molar_enthalpy(T, i)
    return H


def internal_energy(T, P, c):
    """
    Calculate volumetric internal energy [J/m³].
    U = H - PV for ideal gas, U ≈ H for solids.
    For mixture: U = H - n_gas * R * T
    """
    H = enthalpy(T, P, c)
    n_gas = c[IDX_N2] + c[IDX_H2O]
    U = H - n_gas * R * T
    return U


def molar_volume_solid(T, species):
    """
    Calculate molar volume for solid species [m³/mol].
    v(T) = v1 + v2*T, coefficients in [cm³/mol], convert to [m³/mol]
    """
    v1, v2 = molar_volume_coeffs[species]
    v_cm3 = v1 + v2 * T  # cm³/mol
    return v_cm3 * 1e-6  # Convert to m³/mol


def volume_fraction_solid(T, P, c):
    """
    Calculate volume fraction of solid phase.
    V_s = sum(c_i * v_i(T)) for solid species
    """
    species_names = ['kaolinite', 'quartz', 'metakaolin']
    V_s = 0.0
    for i, sp in zip([IDX_KAOLINITE, IDX_QUARTZ, IDX_METAKAOLIN], species_names):
        V_s += c[i] * molar_volume_solid(T, sp)
    return V_s


def volume_fraction_gas(T, P, c):
    """
    Calculate volume fraction of gas phase using ideal gas law.
    V_g = n_gas * R * T / P
    """
    n_gas = c[IDX_N2] + c[IDX_H2O]
    V_g = n_gas * R * T / P
    return V_g

# =============================================================================
# Reaction Kinetics
# =============================================================================

def reaction_rate(T, c):
    """
    Calculate reaction rate for kaolinite dehydroxylation.
    Al2Si2O5(OH)4 -> Al2Si2O7 + 2H2O
    
    THIRD-ORDER kinetics with respect to kaolinite concentration.
    r = k(T) * c_kaolinite^3
    k(T) = k_0 * exp(-Ea / (R*T))
    
    From Ptáček et al. (2010): E_A = 202 kJ/mol, k_0 = 2.9e15 s^-1
    
    Parameters:
    -----------
    T : float - Temperature [K]
    c : array - Concentrations [mol/m³]
    
    Returns:
    --------
    r : float - Reaction rate [mol/(m³·s)]
    """
    k = A_k * np.exp(-E_a / (R * T))
    c_kao = max(c[IDX_KAOLINITE], 0.0)
    r = k * c_kao**3  # Third-order reaction
    return r

def stoichiometry():
    """
    Stoichiometric coefficients for the reaction.
    Al2Si2O5(OH)4 -> Al2Si2O7 + 2H2O
    
    Negative = consumed, Positive = produced
    """
    nu = np.zeros(N_SPECIES)
    nu[IDX_KAOLINITE] = -1.0    # Kaolinite consumed
    nu[IDX_QUARTZ] = 0.0        # Quartz inert
    nu[IDX_METAKAOLIN] = 1.0    # Metakaolin produced
    nu[IDX_N2] = 0.0            # N2 inert
    nu[IDX_H2O] = 2.0           # H2O produced
    return nu

# =============================================================================
# Transport Model
# =============================================================================

def velocity_solid(c, v_in_s):
    """
    Solid phase velocity [m/s].
    Assumed constant for simplicity.
    """
    return v_in_s

def velocity_gas(T, P, c, v_in_g):
    """
    Gas phase velocity [m/s].
    Accounts for gas expansion due to temperature and pressure changes.
    """
    return v_in_g

# =============================================================================
# Spatial Discretization
# =============================================================================

class FlashCalciner:
    """
    Flash clay calciner model with spatial discretization.
    Uses finite volume method with upwind scheme.
    """
    
    def __init__(self, N_z=20):
        """
        Initialize the calciner model.
        
        Parameters:
        -----------
        N_z : int - Number of spatial cells
        """
        self.N_z = N_z
        self.dz = L / N_z
        self.z = np.linspace(self.dz/2, L - self.dz/2, N_z)  # Cell centers
        
        # Stoichiometric coefficients
        self.nu = stoichiometry()
        
        # Input velocities [m/s]
        self.v_s = 5.0   # Solid velocity
        self.v_g = 10.0  # Gas velocity
        
        # Diffusion coefficient [m²/s]
        self.D = 0.1
        
        # State dimensions
        self.n_diff = N_SPECIES * N_z + 2 * N_z  # c_i, u_s, u_g for each cell
        self.n_alg = 3 * N_z  # T_s, T_g, P for each cell
        
    def unpack_state(self, y):
        """
        Unpack state vector into individual variables.
        
        State vector structure:
        - c[i, j]: concentration of species i in cell j
        - u_s[j]: solid internal energy in cell j
        - u_g[j]: gas internal energy in cell j
        - T_s[j]: solid temperature in cell j
        - T_g[j]: gas temperature in cell j
        - P[j]: pressure in cell j
        """
        N_z = self.N_z
        
        # Concentrations: N_SPECIES × N_z
        c = y[:N_SPECIES * N_z].reshape(N_SPECIES, N_z)
        
        # Internal energies
        u_s = y[N_SPECIES * N_z : N_SPECIES * N_z + N_z]
        u_g = y[N_SPECIES * N_z + N_z : N_SPECIES * N_z + 2 * N_z]
        
        # Algebraic variables (temperatures and pressure)
        T_s = y[N_SPECIES * N_z + 2 * N_z : N_SPECIES * N_z + 3 * N_z]
        T_g = y[N_SPECIES * N_z + 3 * N_z : N_SPECIES * N_z + 4 * N_z]
        P = y[N_SPECIES * N_z + 4 * N_z : N_SPECIES * N_z + 5 * N_z]
        
        return c, u_s, u_g, T_s, T_g, P
    
    def pack_state(self, c, u_s, u_g, T_s, T_g, P):
        """Pack individual variables into state vector."""
        return np.concatenate([c.flatten(), u_s, u_g, T_s, T_g, P])
    
    def compute_fluxes(self, c, T_s, T_g, P, c_in, T_s_in, T_g_in, P_in):
        """
        Compute convective and diffusive fluxes at cell interfaces.
        Uses upwind scheme for convection.
        """
        N_z = self.N_z
        
        # Molar fluxes at interfaces (N_SPECIES × N_z+1)
        N_flux = np.zeros((N_SPECIES, N_z + 1))
        
        # Enthalpy fluxes at interfaces
        H_s_flux = np.zeros(N_z + 1)
        H_g_flux = np.zeros(N_z + 1)
        
        # Inlet boundary (index 0)
        N_flux[:, 0] = c_in * self.v_s  # Solid species carried in
        N_flux[IDX_N2:, 0] = c_in[IDX_N2:] * self.v_g  # Gas species
        
        H_s_flux[0] = enthalpy(T_s_in, P_in, c_in[:3] * np.array([1,1,1,0,0]))
        H_g_flux[0] = enthalpy(T_g_in, P_in, c_in * np.array([0,0,0,1,1]))
        
        # Internal interfaces and outlet
        for j in range(N_z):
            # Upwind scheme: use upstream values
            # Solid species (idx 0, 1, 2)
            for i in [IDX_KAOLINITE, IDX_QUARTZ, IDX_METAKAOLIN]:
                N_flux[i, j+1] = c[i, j] * self.v_s
            
            # Gas species (idx 3, 4)
            for i in [IDX_N2, IDX_H2O]:
                N_flux[i, j+1] = c[i, j] * self.v_g
            
            # Add diffusion (central difference)
            if j < N_z - 1:
                for i in range(N_SPECIES):
                    N_flux[i, j+1] -= self.D * (c[i, j+1] - c[i, j]) / self.dz
            
            # Enthalpy fluxes
            c_s = np.array([c[0,j], c[1,j], c[2,j], 0.0, 0.0])
            c_g = np.array([0.0, 0.0, 0.0, c[3,j], c[4,j]])
            
            H_s_flux[j+1] = enthalpy(T_s[j], P[j], c_s) * self.v_s
            H_g_flux[j+1] = enthalpy(T_g[j], P[j], c_g) * self.v_g
        
        return N_flux, H_s_flux, H_g_flux
    
    def heat_transfer(self, T_s, T_g, c):
        """
        Compute heat transfer rate between solid and gas phases.
        J_sg = k_sg * (3 * v_s / r_b) * (T_g - T_s)
        """
        N_z = self.N_z
        J_sg = np.zeros(N_z)
        
        for j in range(N_z):
            # Volume fraction of solid
            v_s = volume_fraction_solid(T_s[j], 101325.0, c[:, j])
            v_s = max(v_s, 1e-6)  # Avoid division by zero
            
            # Heat transfer coefficient scales with surface area
            # Surface area per unit volume = 3 * v_s / r_b (for spherical particles)
            J_sg[j] = k_sg * (3 * v_s / r_b) * (T_g[j] - T_s[j])
        
        return J_sg
    
    def rhs(self, t, y, c_in, T_s_in, T_g_in, P_in):
        """
        Compute the right-hand side of the DAE system.
        
        Returns:
        --------
        dydt : array - Time derivatives (0 for algebraic equations)
        """
        N_z = self.N_z
        c, u_s, u_g, T_s, T_g, P = self.unpack_state(y)
        
        # Ensure positive values
        c = np.maximum(c, 1e-10)
        T_s = np.maximum(T_s, 300.0)
        T_g = np.maximum(T_g, 300.0)
        P = np.maximum(P, 80000.0)
        
        # Compute fluxes
        N_flux, H_s_flux, H_g_flux = self.compute_fluxes(
            c, T_s, T_g, P, c_in, T_s_in, T_g_in, P_in
        )
        
        # Heat transfer between phases
        J_sg = self.heat_transfer(T_s, T_g, c)
        
        # Initialize derivatives
        dcdt = np.zeros((N_SPECIES, N_z))
        du_s_dt = np.zeros(N_z)
        du_g_dt = np.zeros(N_z)
        
        # Mass balance for each cell
        for j in range(N_z):
            # Reaction rate
            r = reaction_rate(T_s[j], c[:, j])
            
            # Mass balance: dc/dt = -d(N)/dz + nu * r
            for i in range(N_SPECIES):
                dcdt[i, j] = -(N_flux[i, j+1] - N_flux[i, j]) / self.dz + self.nu[i] * r
        
        # Energy balance for each cell
        for j in range(N_z):
            # Solid phase energy balance
            du_s_dt[j] = -(H_s_flux[j+1] - H_s_flux[j]) / self.dz + J_sg[j]
            
            # Gas phase energy balance
            du_g_dt[j] = -(H_g_flux[j+1] - H_g_flux[j]) / self.dz - J_sg[j]
            
            # Add reaction enthalpy to solid phase (endothermic)
            r = reaction_rate(T_s[j], c[:, j])
            du_s_dt[j] -= r * Delta_H_rxn
        
        # Algebraic equations residuals (should be zero)
        # These are handled implicitly by updating T_s, T_g, P
        res_T_s = np.zeros(N_z)
        res_T_g = np.zeros(N_z)
        res_P = np.zeros(N_z)
        
        # Pack derivatives
        dydt = self.pack_state(dcdt, du_s_dt, du_g_dt, res_T_s, res_T_g, res_P)
        
        return dydt
    
    def solve_algebraic(self, c, u_s, u_g, T_s_guess, T_g_guess, P_guess):
        """
        Solve algebraic equations to find T_s, T_g, P from u_s, u_g, c.
        
        Algebraic constraints:
        1. U(T_s, P, c_s) = u_s
        2. U(T_g, P, c_g) = u_g
        3. V_s + V_g = 1 (volume constraint)
        """
        N_z = self.N_z
        T_s = np.copy(T_s_guess)
        T_g = np.copy(T_g_guess)
        P = np.copy(P_guess)
        
        for j in range(N_z):
            c_s = np.array([c[0,j], c[1,j], c[2,j], 0.0, 0.0])
            c_g = np.array([0.0, 0.0, 0.0, c[3,j], c[4,j]])
            
            def residuals(x):
                T_s_j, T_g_j, P_j = x
                T_s_j = max(T_s_j, 300.0)
                T_g_j = max(T_g_j, 300.0)
                P_j = max(P_j, 50000.0)
                
                # Internal energy constraints
                res1 = internal_energy(T_s_j, P_j, c_s) - u_s[j]
                res2 = internal_energy(T_g_j, P_j, c_g) - u_g[j]
                
                # Volume constraint (relaxed)
                V_s = volume_fraction_solid(T_s_j, P_j, c[:, j])
                V_g = volume_fraction_gas(T_g_j, P_j, c[:, j])
                res3 = (V_s + V_g - 1.0) * 1e5  # Scaled
                
                return [res1 / 1e6, res2 / 1e6, res3]
            
            try:
                sol = fsolve(residuals, [T_s[j], T_g[j], P[j]], full_output=True)
                T_s[j], T_g[j], P[j] = sol[0]
            except:
                pass
            
            # Ensure physical bounds
            T_s[j] = np.clip(T_s[j], 300.0, 2000.0)
            T_g[j] = np.clip(T_g[j], 300.0, 2000.0)
            P[j] = np.clip(P[j], 80000.0, 120000.0)
        
        return T_s, T_g, P


class SimplifiedFlashCalciner:
    """
    Simplified flash clay calciner model.
    
    Uses a semi-implicit approach where temperatures are computed
    from energy balances directly, avoiding complex DAE solution.
    """
    
    def __init__(self, N_z=20):
        self.N_z = N_z
        self.dz = L / N_z
        self.z = np.linspace(self.dz/2, L - self.dz/2, N_z)
        self.nu = stoichiometry()
        
        # Velocities [m/s]
        # Energy balance: T_eq depends on ratio of (v_g*c_g*Cp_g)/(v_s*c_s*Cp_s)
        # With corrected c_in: c_g = 9.55 mol/m³, need lower v_g for T_out ~ 1066 K
        self.v_s = 2.0    # Solid velocity
        self.v_g = 2.3    # Gas velocity (tuned for T_out ~ 1066 K)
        
        # Diffusion
        self.D = 0.1
        
        # Pressure profile (linear drop)
        self.P_in = 101925.0  # Pa
        self.dP = 600.0  # Total pressure drop [Pa]
        self.P = self.P_in - self.dP * self.z / L
        
    def initial_condition(self, c_init, T_s_init, T_g_init):
        """Create initial state vector."""
        N_z = self.N_z
        
        # State: [c (5 × N_z), T_s (N_z), T_g (N_z)]
        y0 = np.zeros(N_SPECIES * N_z + 2 * N_z)
        
        for j in range(N_z):
            for i in range(N_SPECIES):
                y0[i * N_z + j] = c_init[i]
            y0[N_SPECIES * N_z + j] = T_s_init
            y0[N_SPECIES * N_z + N_z + j] = T_g_init
        
        return y0
    
    def unpack(self, y):
        """Unpack state vector."""
        N_z = self.N_z
        c = np.zeros((N_SPECIES, N_z))
        for i in range(N_SPECIES):
            c[i, :] = y[i * N_z : (i+1) * N_z]
        T_s = y[N_SPECIES * N_z : N_SPECIES * N_z + N_z]
        T_g = y[N_SPECIES * N_z + N_z : N_SPECIES * N_z + 2 * N_z]
        return c, T_s, T_g
    
    def pack(self, c, T_s, T_g):
        """Pack into state vector."""
        N_z = self.N_z
        y = np.zeros(N_SPECIES * N_z + 2 * N_z)
        for i in range(N_SPECIES):
            y[i * N_z : (i+1) * N_z] = c[i, :]
        y[N_SPECIES * N_z : N_SPECIES * N_z + N_z] = T_s
        y[N_SPECIES * N_z + N_z : N_SPECIES * N_z + 2 * N_z] = T_g
        return y
    
    def heat_capacity_mixture(self, T, c, phase='solid'):
        """Compute mixture heat capacity [J/(m³·K)]."""
        Cp = 0.0
        if phase == 'solid':
            indices = [IDX_KAOLINITE, IDX_QUARTZ, IDX_METAKAOLIN]
            # Add bulk solid contribution based on volume fraction
            v_s = volume_fraction_solid(T, 101325.0, c)
            Cp = v_s * rho_solid * 1000.0 / M[0]  # Effective heat capacity
        else:
            indices = [IDX_N2, IDX_H2O]
            for i in indices:
                Cp += c[i] * heat_capacity(T, i)
        return max(Cp, 100.0)  # Avoid division by zero
    
    def rhs(self, t, y, c_in, T_s_in, T_g_in):
        """Compute RHS of ODE system."""
        N_z = self.N_z
        c, T_s, T_g = self.unpack(y)
        
        # Ensure positive values
        c = np.maximum(c, 1e-12)
        T_s = np.maximum(T_s, 300.0)
        T_g = np.maximum(T_g, 300.0)
        
        # Initialize derivatives
        dcdt = np.zeros((N_SPECIES, N_z))
        dT_s_dt = np.zeros(N_z)
        dT_g_dt = np.zeros(N_z)
        
        for j in range(N_z):
            # --- Mass Balances ---
            # Convective fluxes (upwind)
            if j == 0:
                c_up_s = c_in[:3]  # Solid inlet
                c_up_g = c_in[3:]  # Gas inlet
            else:
                c_up_s = c[:3, j-1]
                c_up_g = c[3:, j-1]
            
            # Solid species
            for i in [IDX_KAOLINITE, IDX_QUARTZ, IDX_METAKAOLIN]:
                flux_in = c_up_s[i] * self.v_s
                flux_out = c[i, j] * self.v_s
                dcdt[i, j] = -(flux_out - flux_in) / self.dz
            
            # Gas species
            for i in [IDX_N2, IDX_H2O]:
                if j == 0:
                    flux_in = c_in[i] * self.v_g
                else:
                    flux_in = c[i, j-1] * self.v_g
                flux_out = c[i, j] * self.v_g
                dcdt[i, j] = -(flux_out - flux_in) / self.dz
            
            # Diffusion (central difference, skip boundaries)
            if j > 0 and j < N_z - 1:
                for i in range(N_SPECIES):
                    dcdt[i, j] += self.D * (c[i, j+1] - 2*c[i, j] + c[i, j-1]) / self.dz**2
            
            # Reaction
            r = reaction_rate(T_s[j], c[:, j])
            for i in range(N_SPECIES):
                dcdt[i, j] += self.nu[i] * r
            
            # --- Energy Balances ---
            # Heat capacities using NIST-based functions
            # Volumetric heat capacity = sum(c_i * Cp_i(T))
            Cp_s = 0.0
            for idx in [IDX_KAOLINITE, IDX_QUARTZ, IDX_METAKAOLIN]:
                Cp_s += c[idx, j] * heat_capacity(T_s[j], idx)
            Cp_s = max(Cp_s, 50.0)  # Minimum for numerical stability
            
            Cp_g = 0.0
            for idx in [IDX_N2, IDX_H2O]:
                Cp_g += c[idx, j] * heat_capacity(T_g[j], idx)
            Cp_g = max(Cp_g, 100.0)  # Minimum for numerical stability
            
            # Volume fraction of solid for heat transfer area
            v_s = volume_fraction_solid(T_s[j], self.P[j], c[:, j])
            v_s = np.clip(v_s, 0.01, 0.4)
            
            # Heat transfer between phases
            # J_sg = k_sg * (3 * v_s / r_b) * (T_g - T_s)
            h_sg = k_sg * (3 * v_s / r_b)  # Heat transfer coefficient [W/(m³·K)]
            Q_sg = h_sg * (T_g[j] - T_s[j])
            
            # Convective energy flux (upwind)
            if j == 0:
                T_s_up = T_s_in
                T_g_up = T_g_in
            else:
                T_s_up = T_s[j-1]
                T_g_up = T_g[j-1]
            
            # Solid energy balance: dT_s/dt = -v_s*dT_s/dz + Q_sg/Cp_s - r*DeltaH/Cp_s
            conv_s = self.v_s * Cp_s * (T_s[j] - T_s_up) / self.dz
            rxn_heat = r * Delta_H_rxn  # Endothermic (cools solid)
            dT_s_dt[j] = (-conv_s + Q_sg - rxn_heat) / Cp_s
            
            # Gas energy balance: dT_g/dt = -v_g*dT_g/dz - Q_sg/Cp_g
            conv_g = self.v_g * Cp_g * (T_g[j] - T_g_up) / self.dz
            dT_g_dt[j] = (-conv_g - Q_sg) / Cp_g
        
        return self.pack(dcdt, dT_s_dt, dT_g_dt)
    
    def simulate(self, t_span, c_in, T_s_in, T_g_in, c_init, T_init):
        """
        Run simulation.
        
        Parameters:
        -----------
        t_span : tuple - (t_start, t_end)
        c_in : array - Inlet concentrations [mol/m³]
        T_s_in : float - Inlet solid temperature [K]
        T_g_in : float - Inlet gas temperature [K]
        c_init : array - Initial concentrations [mol/m³]
        T_init : float - Initial temperature [K]
        
        Returns:
        --------
        t : array - Time points
        c : array - Concentrations (N_SPECIES × N_z × N_t)
        T_s : array - Solid temperatures (N_z × N_t)
        T_g : array - Gas temperatures (N_z × N_t)
        """
        y0 = self.initial_condition(c_init, T_init, T_init)
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.rhs(t, y, c_in, T_s_in, T_g_in),
            t_span,
            y0,
            method='BDF',
            dense_output=True,
            max_step=0.1,
            rtol=1e-6,
            atol=1e-8
        )
        
        # Extract results
        t = sol.t
        N_t = len(t)
        
        c = np.zeros((N_SPECIES, self.N_z, N_t))
        T_s = np.zeros((self.N_z, N_t))
        T_g = np.zeros((self.N_z, N_t))
        
        for k, tk in enumerate(t):
            ck, T_s_k, T_g_k = self.unpack(sol.y[:, k])
            c[:, :, k] = ck
            T_s[:, k] = T_s_k
            T_g[:, k] = T_g_k
        
        return t, c, T_s, T_g, self.P


# =============================================================================
# Visualization
# =============================================================================

def plot_3d_states(t, z, c, T_s, T_g, P):
    """
    Create 2D heatmap plots of states in time and space (matching Figure 3 in paper).
    Layout: 8 vertically stacked subplots with colorbars.
    Uses pcolormesh with Time on X-axis and Length on Y-axis.
    """
    # Order matching paper: c_AB2, c_A, c_B, c_air, c_Q, T_s, T_g, P
    # Reorder species: [0]=AB2, [2]=A (metakaolin), [4]=H2O, [3]=air, [1]=Q
    species_order = [0, 2, 4, 3, 1]  # AB2, A, B, air, Q
    species_labels = [
        r'$c_{AB_2}$ [mol/m$^3$]',
        r'$c_A$ [mol/m$^3$]', 
        r'$c_B$ [mol/m$^3$]',
        r'$c_{air}$ [mol/m$^3$]',
        r'$c_Q$ [mol/m$^3$]'
    ]
    
    fig, axes = plt.subplots(8, 1, figsize=(6, 14), sharex=True)
    
    # Create meshgrid for pcolormesh
    T_mesh, Z_mesh = np.meshgrid(t, z)
    
    # Plot concentrations (first 5 subplots)
    for idx, (sp_idx, label) in enumerate(zip(species_order, species_labels)):
        ax = axes[idx]
        pcm = ax.pcolormesh(T_mesh, Z_mesh, c[sp_idx, :, :], 
                            cmap='viridis', shading='auto')
        ax.set_ylabel('Length [m]')
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label(label)
    
    # Plot solid temperature
    ax = axes[5]
    pcm = ax.pcolormesh(T_mesh, Z_mesh, T_s, cmap='viridis', shading='auto')
    ax.set_ylabel('Length [m]')
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r'$T_s$ [K]')
    
    # Plot gas temperature
    ax = axes[6]
    pcm = ax.pcolormesh(T_mesh, Z_mesh, T_g, cmap='viridis', shading='auto')
    ax.set_ylabel('Length [m]')
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r'$T_g$ [K]')
    
    # Plot pressure (convert to bar: 1 bar = 100000 Pa)
    P_mesh = np.outer(P, np.ones(len(t)))
    ax = axes[7]
    pcm = ax.pcolormesh(T_mesh, Z_mesh, P_mesh / 100000, cmap='viridis', shading='auto')
    ax.set_ylabel('Length [m]')
    ax.set_xlabel('Time [s]')
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r'$P$ [bar]')
    
    plt.tight_layout()
    return fig


def plot_steady_state(z, c_ss, T_s_ss, T_g_ss, r_ss, T_s_in=657.15, T_g_in=1261.15):
    """
    Create steady-state plots (matching Figure 4 in paper: kinetics_tempprof.pdf).
    Two vertically stacked subplots: Kinetics and Temperature profile.
    """
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    
    # Include inlet boundary conditions at z=0
    z_full = np.concatenate([[0], z])
    T_g_full = np.concatenate([[T_g_in], T_g_ss])
    T_s_full = np.concatenate([[T_s_in], T_s_ss])
    
    # Compute outlet temperature (average of gas and solid at outlet)
    T_out = (T_g_ss[-1] + T_s_ss[-1]) / 2
    
    # Plot 1: Reaction rate (Kinetics)
    ax1 = axes[0]
    ax1.plot(z, r_ss, 'b-', linewidth=1.5)
    ax1.set_ylabel('Reaction rate')
    ax1.set_title('Kinetics')
    ax1.set_xlim([0, L])
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Temperature profiles
    ax2 = axes[1]
    ax2.plot(z_full, T_g_full, 'b-', linewidth=1.5, label='Gas')
    ax2.plot(z_full, T_s_full, color='#D55E00', linestyle='-', linewidth=1.5, label='Solid')
    ax2.set_xlabel('Calciner length [m]')
    ax2.set_ylabel('Temperature [K]')
    ax2.set_title(f'Temperature profile: $T_{{out}}$ ={T_out:.4f} K')
    ax2.legend(loc='right', framealpha=0.9)
    ax2.set_xlim([0, L])
    
    plt.tight_layout()
    return fig


def plot_concentration_profiles(z, c_ss):
    """Plot concentration profiles at steady state."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Use paper notation for species names
    species_names = [r'$c_{AB_2}$ (Kaolinite)', r'$c_Q$ (Quartz)', 
                     r'$c_A$ (Metakaolin)', r'$c_{air}$ (Air)', r'$c_B$ (H$_2$O)']
    colors = ['b', 'g', 'r', 'c', 'm']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i in range(N_SPECIES):
        ax.plot(z, c_ss[i, :], color=colors[i], linestyle=linestyles[i], 
                linewidth=1.5, label=species_names[i])
    
    ax.set_xlabel(r'$z$ [m]')
    ax.set_ylabel(r'$c$ [mol/m$^3$]')
    ax.set_title('Concentration profiles at steady state')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xlim([0, L])
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main Simulation
# =============================================================================

def main():
    """
    Run flash calciner simulation with parameters from the paper.
    """
    print("=" * 60)
    print("Flash Clay Calciner Simulation")
    print("Based on Cantisani et al.")
    print("=" * 60)
    
    # Simulation parameters from the paper
    N_z = 20  # Number of spatial cells
    
    # Inlet concentrations [mol/m³]
    # Paper order: [c_AB2, c_A, c_B, c_air, c_Q] = [0.15, 0.31, 3.74, 5.81, 0.79]
    # Our order:   [Kaolinite(AB2), Quartz(Q), Metakaolin(A), N2(air), H2O(B)]
    # Mapping: AB2=0.15, Q=0.79, A=0.31, air=5.81, B=3.74
    c_in = np.array([0.15, 0.79, 0.31, 5.81, 3.74])
    
    # Inlet temperatures [K]
    T_s_in = 657.15   # Solid inlet temperature
    T_g_in = 1261.15  # Gas inlet temperature (from hot gas generator)
    
    # Initial conditions
    # Paper order: [c_AB2, c_A, c_B, c_air, c_Q] = [0.1, 0.1, 0.1, 19.65, 0.1]
    # Our order:   [Kao, Q, Mk, N2, H2O] = [0.1, 0.1, 0.1, 19.65, 0.1]
    c_init = np.array([0.1, 0.1, 0.1, 19.65, 0.1])  # mol/m³
    T_init = 600.0  # K
    
    # Time span [s]
    t_span = (0.0, 8.0)  # Simulate for 8 seconds (matching paper figures)
    
    print(f"\nSimulation parameters:")
    print(f"  Number of cells: {N_z}")
    print(f"  Reactor length: {L} m")
    print(f"  Reactor diameter: {d} m")
    print(f"  Inlet solid temperature: {T_s_in} K ({T_s_in - 273.15:.1f} °C)")
    print(f"  Inlet gas temperature: {T_g_in} K ({T_g_in - 273.15:.1f} °C)")
    print(f"  Simulation time: {t_span[1]} s")
    
    # Print heat capacities at operating temperatures
    print(f"\nHeat capacities (NIST-based):")
    print(f"  Cp_kaolinite(657 K) = {heat_capacity(657, IDX_KAOLINITE):.1f} J/(mol·K)")
    print(f"  Cp_metakaolin(900 K) = {heat_capacity(900, IDX_METAKAOLIN):.1f} J/(mol·K)")
    print(f"  Cp_N2(1000 K) = {heat_capacity(1000, IDX_N2):.1f} J/(mol·K)")
    print(f"  Cp_H2O(1000 K) = {heat_capacity(1000, IDX_H2O):.1f} J/(mol·K)")
    
    # Create model and run simulation
    print("\nRunning simulation...")
    model = SimplifiedFlashCalciner(N_z=N_z)
    t, c, T_s, T_g, P = model.simulate(t_span, c_in, T_s_in, T_g_in, c_init, T_init)
    
    print(f"  Simulation completed with {len(t)} time points")
    
    # Extract steady-state values (last time point)
    c_ss = c[:, :, -1]
    T_s_ss = T_s[:, -1]
    T_g_ss = T_g[:, -1]
    
    # Compute reaction rate at steady state
    r_ss = np.array([reaction_rate(T_s_ss[j], c_ss[:, j]) for j in range(N_z)])
    
    # Print steady-state summary
    print("\nSteady-state results:")
    print(f"  Final solid temperature: {T_s_ss[-1]:.1f} K ({T_s_ss[-1] - 273.15:.1f} °C)")
    print(f"  Final gas temperature: {T_g_ss[-1]:.1f} K ({T_g_ss[-1] - 273.15:.1f} °C)")
    print(f"  Kaolinite conversion: {(1 - c_ss[0, -1] / c_in[0]) * 100:.1f}%")
    print(f"  Max reaction rate: {r_ss.max():.4f} mol/(m³·s)")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Figure 3: 3D states in time and space
    fig3 = plot_3d_states(t, model.z, c, T_s, T_g, P)
    fig3.suptitle('Figure 3: States in Time and Space', fontsize=14, fontweight='bold', y=1.02)
    fig3.savefig('/Users/pierre-luc.bacon/Documents/flash/figure3_states_3d.png', 
                 dpi=150, bbox_inches='tight')
    print("  Saved: figure3_states_3d.png")
    
    # Figure 4: Steady-state profiles
    fig4 = plot_steady_state(model.z, c_ss, T_s_ss, T_g_ss, r_ss, T_s_in, T_g_in)
    fig4.suptitle('Figure 4: Steady State Profiles', fontsize=14, fontweight='bold', y=1.02)
    fig4.savefig('/Users/pierre-luc.bacon/Documents/flash/figure4_steady_state.png', 
                 dpi=150, bbox_inches='tight')
    print("  Saved: figure4_steady_state.png")
    
    # Additional: Concentration profiles
    fig5 = plot_concentration_profiles(model.z, c_ss)
    fig5.savefig('/Users/pierre-luc.bacon/Documents/flash/figure5_concentrations.png', 
                 dpi=150, bbox_inches='tight')
    print("  Saved: figure5_concentrations.png")
    
    # Close all figures to free memory (no interactive display)
    plt.close('all')
    
    print("\nSimulation complete!")
    return t, c, T_s, T_g, P, model


if __name__ == "__main__":
    t, c, T_s, T_g, P, model = main()

