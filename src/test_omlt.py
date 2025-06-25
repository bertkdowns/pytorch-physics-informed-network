import pyomo.environ as pyo
import torch
import onnxruntime as ort
from omlt_integration import create_heat_exchanger_omlt_model

def test_omlt():
    model = create_heat_exchanger_omlt_model("models/heat_exchanger_pinn.onnx")

    # # Print model structure before solving
    # print("=== MODEL STRUCTURE ===")
    # model.pprint()
    
    # Example: add a trivial objective to see if it solves
    # In practice, you would define a meaningful objective based on your problem
    # such as minimising energy usage or maximising heat transfer
    model.obj = pyo.Objective(expr=(model.m_hot - 0.18)**2)

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(model, tee=False)
    print(result)

    print("Solved model:")
    print(f"  m_hot = {pyo.value(model.m_hot)} kg/s")
    print(f"  m_cold = {pyo.value(model.m_cold)} kg/s")
    print(f"  T_hot_out = {pyo.value(model.T_hot_out)} K")
    print(f"  T_cold_out = {pyo.value(model.T_cold_out)} K")

    print("\n=== VERIFICATION WITH PYTORCH/ONNX ===")
    # Get the solution values
    m_hot = pyo.value(model.m_hot)
    m_cold = pyo.value(model.m_cold)
    T_hot_in = pyo.value(model.T_hot_in)
    T_cold_in = pyo.value(model.T_cold_in)
    
    # Create input tensor
    test_input = torch.tensor([[m_hot, m_cold, T_hot_in, T_cold_in]], dtype=torch.float32)
    
    # Run the ONNX model directly
    ort_session = ort.InferenceSession("models/heat_exchanger_pinn.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print("Direct ONNX prediction:")
    print(f"  T_hot_out = {ort_outputs[0][0][0]:.6f} K")
    print(f"  T_cold_out = {ort_outputs[0][0][1]:.6f} K")
    
    print("\nOMLT/Pyomo result:")
    print(f"  T_hot_out = {pyo.value(model.T_hot_out):.6f} K")
    print(f"  T_cold_out = {pyo.value(model.T_cold_out):.6f} K")
    
    print(f"Difference: {abs(ort_outputs[0][0][0] - pyo.value(model.T_hot_out)):.6f} K")

if __name__ == '__main__':
    test_omlt()