import onnx
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io import load_onnx_neural_network

def create_heat_exchanger_omlt_model(onnx_path="models/heat_exchanger_pinn.onnx"):
    """
    Create a Pyomo model that wraps the ONNX neural network.
    This allows the neural network to be used in optimisation problems via OMLT.
    """

    # Load the ONNX file from the specified path
    onnx_model = onnx.load(onnx_path)

    # Parse the ONNX model into a NetworkDefinition
    # Provide optional input bounds for each input index
    # Needed because the ONNX model does not have input bounds defined
    input_bounds = {
        0: (0.144, 0.216),  # m_dot_hot (kg/s)
        1: (0.108, 0.180),  # m_dot_cold (kg/s)
        2: (370.0, 410.0),  # T_hot_in (K)
        3: (290.0, 310.0)   # T_cold_in (K)
    }
    net = load_onnx_neural_network(onnx_model, input_bounds=input_bounds)

    # Create a Pyomo model and add an OMLT block
    model = pyo.ConcreteModel()
    model.nn_block = OmltBlock()

    # Select a formulation and build the neural network representation
    formulation = FullSpaceNNFormulation(net)
    model.nn_block.build_formulation(formulation)

    # Define Pyomo variables for the inputs
    model.m_hot = pyo.Var(bounds=input_bounds[0])    # mass flow hot
    model.m_cold = pyo.Var(bounds=input_bounds[1])   # mass flow cold
    model.T_hot_in = pyo.Var(bounds=input_bounds[2]) # inlet temp hot
    model.T_cold_in = pyo.Var(bounds=input_bounds[3]) # inlet temp cold

    # Define Pyomo variables for the model outputs
    model.T_hot_out = pyo.Var()
    model.T_cold_out = pyo.Var()

    # Connect the OMLT inputs
    @model.Constraint()
    def connect_m_hot(mm):
        return mm.m_hot == mm.nn_block.inputs[0]

    @model.Constraint()
    def connect_m_cold(mm):
        return mm.m_cold == mm.nn_block.inputs[1]

    @model.Constraint()
    def connect_T_hot_in(mm):
        return mm.T_hot_in == mm.nn_block.inputs[2]

    @model.Constraint()
    def connect_T_cold_in(mm):
        return mm.T_cold_in == mm.nn_block.inputs[3]

    # Connect the OMLT outputs
    @model.Constraint()
    def connect_T_hot_out(mm):
        return mm.T_hot_out == mm.nn_block.outputs[0]

    @model.Constraint()
    def connect_T_cold_out(mm):
        return mm.T_cold_out == mm.nn_block.outputs[1]

    # Now m is a standard Pyomo model that includes the neural network
    # Can add additional constraints, an objective, etc.
    return model
    
