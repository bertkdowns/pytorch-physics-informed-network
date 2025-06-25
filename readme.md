# Pytorch physics informed neural network.


## Goals of this repository:

- Figure out how to use pytorch
- Figure out how to add soft constraints (into the loss function) to enforce the physics
- Figure out how to use meta-learning


Initial result based on [Li et al., Digital twins for accurate prediction beyond routine operation,](https://doi.org/10.1016/j.compchemeng.2025.109211.)

- Exporting a model in ONNX format
- Importing the model into IDAES using OMLT

# Stretch Goals: Surrogate Informed Neural Network

- create a model in IDAES
- use raw data for parameter estimation of unknowns in model (e.g for heat transfer coefficient on a heat exchanger). This creates a first principles representaiton
- use the first principles idaes model (with the estimated parameters) to generate a bunch of synthetic data (including data outside the training region)
- create neural network structure
- train the neural network based on the synthetic data (making a surrogate model)
- fine-tune the neural network based on the real data.

Outlined in https://bertkdowns.github.io/thesis/notes/gug7rrgwx6lsajhmhfr56w0/

---

## Repository Overview

This repository provides a full workflow for building, training, validating, and deploying a **Physics-Informed Neural Network (PINN)** for heat exchanger modeling, with seamless integration into process optimisation frameworks using OMLT and Pyomo. The codebase is designed for practical engineering applications, enabling data-driven modeling that respects physical laws and can be embedded in larger optimisation problems.

### Purpose

- **Data-Driven Surrogate Modeling:** Create neural network surrogates for process units (here, a heat exchanger) using synthetic data generated from first-principles models (IDAES).
- **Physics-Informed Learning:** Enforce physical constraints (e.g., energy conservation) as soft constraints in the neural network loss function.
- **Meta-Learning:** Dynamically adapt the weighting of physics constraints during training for optimal balance between data fit and physical realism.
- **Optimisation Integration:** Export trained models to ONNX and use OMLT to embed them in Pyomo optimisation problems.

---

## Codebase Structure

- **src/generate_heat_exchanger_data.py**  
  Generates synthetic data using the IDAES process modeling framework. Uses Latin Hypercube Sampling (LHS) for efficient coverage of the parameter space. Outputs are saved as PyTorch tensors for training.

- **src/heat_exchanger_physics.py**  
  Defines the physics residual (energy balance) for the heat exchanger, used as a soft constraint in the PINN loss function.

- **src/residual_network.py**  
  Implements the residual neural network (rMLP) architecture, training loop, meta-learning for physics weights, and ONNX export. Trains the PINN using both empirical and physics-based losses.

- **src/omlt_integration.py**  
  Loads the ONNX model, defines input bounds, and builds a Pyomo model using OMLT. Connects Pyomo variables to the neural network for use in optimisation.

- **src/test_onnx_model.py**  
  Validates that the exported ONNX model produces the same outputs as the original PyTorch model for a set of test cases.

- **src/test_omlt.py**  
  Demonstrates loading the ONNX model into Pyomo/OMLT, adding an objective, and solving with Ipopt. Compares OMLT/Pyomo results to direct ONNX inference.

- **run_pinn.sh**  
  Automates the workflow: data generation, model training, and ONNX export.

- **test_onnx.sh**  
  Checks ONNX/PyTorch consistency.

- **test_omlt.sh**  
  Runs the OMLT/Pyomo integration test.

---

## Workflow

1. **Data Generation**  
   - Synthetic data is generated using IDAES with Latin Hypercube Sampling for the input parameters (hot/cold flow rates, inlet temperatures).
   - The data is saved as `data/heat_exchanger_inputs.pt` and `data/heat_exchanger_outputs.pt`.
   - A visualisation of the LHS sampling is saved as `data/lhs_sampling_visualisation.png`.

2. **Model Training**  
   - The PINN is trained using both empirical loss (data fit) and a physics-based loss (energy balance).
   - Meta-learning dynamically adjusts the weight of the physics constraint.
   - The trained model is saved as `models/heat_exchanger_pinn_model.pt` and exported to ONNX as `models/heat_exchanger_pinn.onnx`.

3. **Model Validation**  
   - `test_onnx.sh` and `test_onnx_model.py` ensure the ONNX model matches PyTorch predictions.
   - `test_omlt.sh` and `test_omlt.py` load the ONNX model into Pyomo/OMLT, solve a simple optimisation, and compare results to direct ONNX inference.

4. **Optimisation Integration**  
   - The ONNX model is embedded in a Pyomo model using OMLT, allowing for optimisation and parameter studies with the neural network as a constraint.

---

## How to Run

1. **Train and Export the Model**
   ```bash
   ./run_pinn.sh
   ```
   - Generates synthetic data and trains the PINN.
   - Exports the trained model to ONNX.

2. **Verify ONNX Consistency**
   ```bash
   ./test_onnx.sh
   ```
   - Compares PyTorch and ONNX model predictions for several test cases.

3. **Verify OMLT Integration**
   ```bash
   ./test_omlt.sh
   ```
   - Loads the ONNX model into Pyomo/OMLT, solves an optimisation problem, and prints results.

---

## What to Expect

- **Data Generation:**  
  - Console output showing each synthetic data point generated by IDAES.
  - LHS sampling visualisation in `data/lhs_sampling_visualisation.png`.

- **Training:**  
  - Progress logs showing loss values, meta-learning weight adaptation, and convergence.
  - Final test prediction for a sample input.

- **ONNX Validation:**  
  - Side-by-side comparison of PyTorch and ONNX predictions, with differences typically <1e-5.

- **OMLT/Pyomo Integration:**  
  - Solver output showing successful optimisation with the neural network surrogate.
  - Comparison of OMLT/Pyomo and direct ONNX predictions.

---

## Features

- **Physics-Informed Loss:**  
  Enforces energy conservation in the heat exchanger model.

- **Meta-Learning:**  
  Dynamically tunes the importance of the physics constraint during training.

- **Efficient Sampling:**  
  Uses Latin Hypercube Sampling for better parameter space coverage.

- **ONNX Export and OMLT Integration:**  
  Enables deployment of the trained PINN in process optimisation frameworks.

- **Validation and Testing:**  
  Includes scripts to ensure model consistency and correct integration.

---

## Example Output

```
===== Physics-Informed Neural Network Workflow =====
[1/2] Generating heat exchanger data...
[1/20] Input=(0.144, 0.160, 393.52, 306.86) => Output=(373.14, 373.16)
...
[20/20] Input=(0.145, 0.150, 392.67, 304.13) => Output=(373.14, 373.16)
Generated 20 data points.
Saved X to data/heat_exchanger_inputs.pt and y to data/heat_exchanger_outputs.pt.

[2/2] Training physics-informed neural network...
Loaded 20 samples for training
Epoch 0: Total Loss=559058.375000, Empirical=208812.625000, PDE=350245.781250, w=64.975418
...
Epoch 1999: Total Loss=61.660347, Empirical=52.121796, PDE=953.855103, w=1.000000

Test prediction:
Input: m_hot=0.180 kg/s, m_cold=0.150 kg/s
       T_hot_in=390.0K, T_cold_in=295.0K
Predicted: T_hot_out=369.84K, T_cold_out=369.97K
Model saved to models/heat_exchanger_pinn_model.pt
```

---

## Additional Notes

- **Warnings during OMLT/Pyomo runs** (e.g., "Setting Var ‘nn_block.scaled_inputs[ ]’…") are harmless and can be ignored.
- The codebase is modular and can be extended to other process units or physical systems by changing the data generation and physics residual definitions.
- For more details on the meta-learning approach and PINN methodology, see the referenced paper and code comments.

---

This repository provides a robust, end-to-end example of combining first-principles simulation, machine learning, and optimisation for process systems engineering.

