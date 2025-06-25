import torch
import numpy as np
import matplotlib.pyplot as plt

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.heat_exchanger import HeatExchanger
from pyomo.environ import ConcreteModel, SolverFactory, value
from property_packages.build_package import build_package
from smt.sampling_methods import LHS


def visualise_samples(samples, xlimits, filename="data/lhs_sampling_visualisation.png"):
    """
    Visualise the distribution of samples across input parameters
    to verify the quality of Latin Hypercube Sampling.
    """
    param_names = ['m_hot (mol/s)', 'm_cold (mol/s)', 'T_hot_in (K)', 'T_cold_in (K)']
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    # Plot all pairwise projections
    plot_idx = 0
    for i in range(4):
        for j in range(i+1, 4):
            ax = axs[plot_idx]
            ax.scatter(samples[:, i], samples[:, j], alpha=0.7)
            ax.set_xlabel(param_names[i])
            ax.set_ylabel(param_names[j])
            ax.grid(True)
            
            # Add bounds from xlimits
            ax.set_xlim([xlimits[i][0], xlimits[i][1]])
            ax.set_ylim([xlimits[j][0], xlimits[j][1]])
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Sampling visualisation saved to {filename}")


def generate_heat_exchanger_data(num_samples=20, visualise=True, criterion='ese'):
    """
    Generate steady-state heat exchanger data using Latin Hypercube Sampling
    to efficiently explore the parameter space of flow rates and inlet temperatures.
    
    Args:
        num_samples: Number of data points to generate
        visualise: Whether to create a visualisation of the sampled points
        criterion: LHS criterion ('ese' is recommended for best space-filling)
    """
    X_data = []  # inputs
    y_data = []  # outputs

    # Ranges to vary
    m_hot_range = (8, 12)        # mol/s (hot side)
    m_cold_range = (6, 10)       # mol/s (cold side)
    T_hot_in_range = (370, 410)  # K
    T_cold_in_range = (290, 310) # K

    # Approx. conversion for water (18 g/mol => 0.018 kg/mol)
    mol_to_kg = 0.018

    # Define parameter space for Latin Hypercube Sampling
    xlimits = np.array([
        m_hot_range,     # m_hot range
        m_cold_range,    # m_cold range
        T_hot_in_range,  # T_hot_in range
        T_cold_in_range  # T_cold_in range
    ])
    
    # Create LHS sampler with enhanced space-filling properties (ESE criterion)
    sampling = LHS(xlimits=xlimits, criterion=criterion, random_state=42)
    
    # Generate samples using Latin Hypercube method
    lhs_samples = sampling(num_samples)
    
    print(f"Generated {num_samples} Latin Hypercube samples with '{criterion}' criterion")
    
    # Visualise the distribution of samples if requested
    if visualise:
        visualise_samples(lhs_samples, xlimits)

    # Process each sample point
    successful_samples = 0
    for i in range(num_samples):
        # Build a new model
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.water_props = build_package("helmholtz", ["water"], ["Liq","Vap"])

        m.fs.hx = HeatExchanger(
            hot_side={"property_package": m.fs.water_props},
            cold_side={"property_package": m.fs.water_props},
        )

        # Extract parameter values from LHS sample
        m_hot = lhs_samples[i, 0]      # mol/s
        m_cold = lhs_samples[i, 1]     # mol/s
        T_hot_in = lhs_samples[i, 2]   # K
        T_cold_in = lhs_samples[i, 3]  # K

        # Fix hot side conditions
        m.fs.hx.hot_side_inlet.flow_mol.fix(m_hot)
        m.fs.hx.hot_side.properties_in[0].constrain_component(
            m.fs.hx.hot_side.properties_in[0].temperature,
            T_hot_in
        )
        m.fs.hx.hot_side_inlet.pressure.fix(101325)  # Pa

        # Fix cold side conditions
        m.fs.hx.cold_side_inlet.flow_mol.fix(m_cold)
        m.fs.hx.cold_side.properties_in[0].constrain_component(
            m.fs.hx.cold_side.properties_in[0].temperature,
            T_cold_in
        )
        m.fs.hx.cold_side_inlet.pressure.fix(101325)  # Pa

        # Additional HX parameters
        m.fs.hx.area.fix(50.0)  # m^2
        m.fs.hx.overall_heat_transfer_coefficient.fix(150.0)  # W/m^2.K

        # Solve the model
        try:
            solver = SolverFactory("ipopt")
            m.fs.hx.initialize(outlvl=0)
            results = solver.solve(m, tee=False)

            if results.solver.termination_condition == "optimal":
                # Convert mol to kg
                m_hot_kg = m_hot * mol_to_kg
                m_cold_kg = m_cold * mol_to_kg

                # Extract the outlet temperatures
                T_hot_out = value(m.fs.hx.hot_side.properties_out[0].temperature)
                T_cold_out = value(m.fs.hx.cold_side.properties_out[0].temperature)

                # Store for training
                X_data.append([m_hot_kg, m_cold_kg, T_hot_in, T_cold_in])
                y_data.append([T_hot_out, T_cold_out])
                successful_samples += 1
                print(f"[{i+1}/{num_samples}] Input=({m_hot_kg:.3f}, {m_cold_kg:.3f}, {T_hot_in:.2f}, {T_cold_in:.2f}) => "
                      f"Output=({T_hot_out:.2f}, {T_cold_out:.2f})")
            else:
                print(f"Solver not optimal at iteration {i+1}, skipping.")
        except Exception as e:
            print(f"IDAES simulation failed at iteration {i+1}: {e}")

    # Convert to Torch tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    # Optionally save to disk
    torch.save(X_tensor, "data/heat_exchanger_inputs.pt")
    torch.save(y_tensor, "data/heat_exchanger_outputs.pt")
    
    print(f"\nGenerated {len(X_tensor)}/{num_samples} valid data points using Latin Hypercube Sampling.")
    return X_tensor, y_tensor


if __name__ == "__main__":
    # You can customize how many samples to generate
    X, y = generate_heat_exchanger_data(num_samples=20, visualise=True, criterion='ese')
    print("Saved X to data/heat_exchanger_inputs.pt and y to data/heat_exchanger_outputs.pt.")