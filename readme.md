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

