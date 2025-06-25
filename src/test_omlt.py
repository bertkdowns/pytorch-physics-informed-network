import pyomo.environ as pyo
from omlt_integration import create_heat_exchanger_omlt_model

def test_omlt():
    model = create_heat_exchanger_omlt_model("models/heat_exchanger_pinn.onnx")
    
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

if __name__ == '__main__':
    test_omlt()