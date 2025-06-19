import torch
import numpy as np

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.heat_exchanger import HeatExchanger
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import ConcreteModel, SolverFactory, value
from property_packages.build_package import build_package


def generate_heat_exchanger_data(num_samples=20):
    """
    Generate steady-state heat exchanger data by varying flow rates and inlet temperatures.
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

    for i in range(num_samples):
        # Build a new model
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.water_props = build_package("helmholtz", ["water"], ["Liq","Vap"])

        m.fs.hx = HeatExchanger(
            hot_side={"property_package": m.fs.water_props},
            cold_side={"property_package": m.fs.water_props},
        )

        # Generate random input values within the specified ranges
        m_hot = np.random.uniform(m_hot_range[0], m_hot_range[1])  # mol/s
        m_cold = np.random.uniform(m_cold_range[0], m_cold_range[1])  # mol/s
        T_hot_in = np.random.uniform(T_hot_in_range[0], T_hot_in_range[1])  # K
        T_cold_in = np.random.uniform(T_cold_in_range[0], T_cold_in_range[1])  # K

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
    
    print(f"\nGenerated {len(X_tensor)} data points.")
    return X_tensor, y_tensor


if __name__ == "__main__":
    # You can customise how many samples to generate
    X, y = generate_heat_exchanger_data(num_samples=20)
    print("Saved X to data/heat_exchanger_inputs.pt and y to data/heat_exchanger_outputs.pt.")