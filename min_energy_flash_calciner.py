import numpy as np
import matplotlib.pyplot as plt

from flash_calciner import (
    SimplifiedFlashCalciner,
    IDX_KAOLINITE,
    IDX_N2,
    IDX_H2O,
    A_cross,
    molar_enthalpy,
)

# -----------------------------------------------------------------------------
# Helpers: quality (conversion) and heater power
# -----------------------------------------------------------------------------

def simulate_steady_state(T_g_in,
                          model,
                          t_span=(0.0, 8.0),
                          c_in=None,
                          T_s_in=657.15,
                          c_init=None,
                          T_init=300.0):
    """
    Run the calciner to (approximate) steady state and return outlet quantities.
    """
    if c_in is None:
        # From your main(): [Kaolinite, Quartz, Metakaolin, N2, H2O]
        c_in = np.array([0.15, 0.79, 0.31, 5.81, 3.74])

    if c_init is None:
        # From your main() initial condition
        c_init = np.array([0.1, 0.1, 0.1, 19.65, 0.1])

    t, c, T_s, T_g, P = model.simulate(t_span, c_in, T_s_in, T_g_in, c_init, T_init)

    # Outlet cell = last spatial index, last time index
    c_out = c[IDX_KAOLINITE, -1, -1]
    alpha = 1.0 - c_out / c_in[IDX_KAOLINITE]  # kaolinite conversion

    T_s_out = T_s[-1, -1]
    T_g_out = T_g[-1, -1]

    # Steady profiles along z
    c_ss = c[:, :, -1].copy()
    T_s_ss = T_s[:, -1].copy()
    T_g_ss = T_g[:, -1].copy()

    return {
        "alpha": alpha,
        "T_s_out": T_s_out,
        "T_g_out": T_g_out,
        "c_ss": c_ss,
        "T_s_ss": T_s_ss,
        "T_g_ss": T_g_ss,
        "z": model.z.copy(),
    }


def heater_power(T_g_in, model, c_in=None, T_cold=300.0):
    """
    Approximate electric heater power [W] needed to raise the gas from T_cold
    to T_g_in, for the inlet gas mixture and gas velocity used in the model.
    """
    if c_in is None:
        c_in = np.array([0.15, 0.79, 0.31, 5.81, 3.74])

    # Molar enthalpy change for N2 and H2O between T_cold and T_g_in
    dH_N2 = molar_enthalpy(T_g_in, IDX_N2) - molar_enthalpy(T_cold, IDX_N2)
    dH_H2O = molar_enthalpy(T_g_in, IDX_H2O) - molar_enthalpy(T_cold, IDX_H2O)

    # Molar flux per unit area [mol/(m²·s)] = v_g * c
    Ndot_N2 = model.v_g * c_in[IDX_N2]
    Ndot_H2O = model.v_g * c_in[IDX_H2O]

    # Power [W] = enthalpy flux * area
    Qdot = A_cross * (Ndot_N2 * dH_N2 + Ndot_H2O * dH_H2O)
    return Qdot


# -----------------------------------------------------------------------------
# Min-energy optimization at fixed throughput / quality
# -----------------------------------------------------------------------------

def optimize_min_energy(
    alpha_min=0.99,
    T_g_min=900.0,
    T_g_max=1300.0,
    n_grid=15,
    t_span=(0.0, 8.0),
):
    """
    Solve:  min P_el(T_g_in)
           s.t. alpha(T_g_in) >= alpha_min

    using a simple grid search over T_g_in.
    """
    # Fixed inlet composition and initial state from your main()
    c_in = np.array([0.15, 0.79, 0.31, 5.81, 3.74])
    c_init = np.array([0.1, 0.1, 0.1, 19.65, 0.1])
    T_s_in = 657.15
    T_init = 300.0

    # Use a fresh model per run (simpler, avoids statefulness)
    def run_for_Tg(Tg):
        model = SimplifiedFlashCalciner(N_z=20)
        ss = simulate_steady_state(
            Tg, model,
            t_span=t_span,
            c_in=c_in,
            T_s_in=T_s_in,
            c_init=c_init,
            T_init=T_init,
        )
        power = heater_power(Tg, model, c_in=c_in)
        return ss, power

    # Baseline at nominal gas inlet temperature from the paper
    T_g_nominal = 1261.15  # K
    ss_base, P_base = run_for_Tg(T_g_nominal)
    alpha_base = ss_base["alpha"]

    print("=== Baseline operation (paper-like) ===")
    print(f"T_g,in = {T_g_nominal:.2f} K")
    print(f"  alpha_out   = {alpha_base*100:.2f} %")
    print(f"  T_s,out     = {ss_base['T_s_out']:.2f} K")
    print(f"  Heater power= {P_base/1e6:.3f} MW\n")

    # If alpha_min is given as a fraction of baseline, you can do:
    # alpha_min = 0.99 * alpha_base

    # Grid over candidate inlet gas temperatures
    T_grid = np.linspace(T_g_min, T_g_max, n_grid)
    results = []

    print("=== Grid search over T_g,in ===")
    for Tg in T_grid:
        ss, P_el = run_for_Tg(Tg)
        alpha = ss["alpha"]

        feasible = alpha >= alpha_min
        print(
            f"T_g,in = {Tg:7.2f} K | "
            f"alpha = {alpha:6.3f} | "
            f"P_el = {P_el/1e6:6.3f} MW | "
            f"{'FEASIBLE' if feasible else 'infeasible'}"
        )

        results.append(
            {
                "T_g_in": Tg,
                "alpha": alpha,
                "P_el": P_el,
                "feasible": feasible,
                "ss": ss,
            }
        )

    # Select best feasible point (min power)
    feasible_points = [r for r in results if r["feasible"]]
    if not feasible_points:
        raise RuntimeError("No feasible T_g,in found for the given alpha_min.")

    best = min(feasible_points, key=lambda r: r["P_el"])

    print("\n=== Optimal min-energy operating point ===")
    print(f"T_g,in*      = {best['T_g_in']:.2f} K")
    print(f"alpha_out*   = {best['alpha']*100:.2f} %")
    print(f"P_el*        = {best['P_el']/1e6:.3f} MW")
    print(f"Energy savings vs baseline: "
          f"{(1.0 - best['P_el']/P_base)*100:.1f} %")

    return {
        "baseline": {"T_g_in": T_g_nominal, "P_el": P_base, "ss": ss_base},
        "best": best,
        "grid": results,
    }


# -----------------------------------------------------------------------------
# Simple visualization
# -----------------------------------------------------------------------------

def plot_results(result):
    """
    Make a couple of quick plots:
      - conversion vs T_g,in
      - heater power vs T_g,in
      - baseline vs optimal temperature profiles along z
    """
    grid = result["grid"]
    baseline = result["baseline"]
    best = result["best"]

    T_grid = np.array([r["T_g_in"] for r in grid])
    alpha_grid = np.array([r["alpha"] for r in grid])
    P_grid = np.array([r["P_el"] for r in grid]) / 1e6  # MW

    # 1) alpha and P_el vs T_g,in
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    ax1, ax2 = axes

    ax1.plot(T_grid, alpha_grid, marker="o")
    ax1.axhline(best["alpha"], color="k", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Kaolinite conversion α_out [-]")
    ax1.grid(True)

    ax2.plot(T_grid, P_grid, marker="o")
    ax2.axvline(best["T_g_in"], color="r", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Inlet gas temperature T_g,in [K]")
    ax2.set_ylabel("Heater power [MW]")
    ax2.grid(True)

    fig.tight_layout()
    fig.suptitle("Min-energy operation at fixed throughput / quality",
                 y=1.02)

    # 2) Profiles along reactor: baseline vs optimal
    z = baseline["ss"]["z"]
    z_full = np.concatenate([[0.0], z])  # include inlet point

    T_s_base = baseline["ss"]["T_s_ss"]
    T_g_base = baseline["ss"]["T_g_ss"]
    T_s_opt = best["ss"]["T_s_ss"]
    T_g_opt = best["ss"]["T_g_ss"]

    fig2, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(z_full, np.concatenate([[657.15], T_s_base]), label="Solid (baseline)")
    ax.plot(z_full, np.concatenate([[657.15], T_s_opt]), label="Solid (optimal)")
    ax.plot(z_full, np.concatenate([[baseline['T_g_in']], T_g_base]),
            label="Gas (baseline)", linestyle="--")
    ax.plot(z_full, np.concatenate([[best['T_g_in']], T_g_opt]),
            label="Gas (optimal)", linestyle="--")

    ax.set_xlabel("Calciner length z [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Steady-state temperature profiles")
    ax.legend()
    ax.grid(True)
    fig2.tight_layout()

    # Save figures
    fig.savefig("min_energy_grid_search.png", dpi=150, bbox_inches="tight")
    fig2.savefig("min_energy_profiles.png", dpi=150, bbox_inches="tight")
    print("\nSaved: min_energy_grid_search.png, min_energy_profiles.png")
    plt.close('all')


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    result = optimize_min_energy(
        alpha_min=0.99,     # required outlet conversion
        T_g_min=900.0,      # search range for T_g,in
        T_g_max=1300.0,
        n_grid=10,          # keep small at first; can refine later
        t_span=(0.0, 8.0),  # same horizon as your paper reproduction
    )
    plot_results(result)
