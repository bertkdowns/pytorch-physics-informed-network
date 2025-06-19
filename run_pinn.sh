#!/bin/bash
# Heat Exchanger PINN Workflow

set -e  # Exit on error

echo "===== Physics-Informed Neural Network Workflow ====="

# Make sure directories exist
mkdir -p data models

# Step 1: Generate data
echo "[1/2] Generating heat exchanger data..."
python generate_heat_exchanger_data.py

# Step 2: Train the model
echo "[2/2] Training physics-informed neural network..."
python residual-network.py

echo "===== Workflow Complete! ====="
echo "Generated data saved to data/ directory"
echo "Trained model saved to models/heat_exchanger_pinn_model.pt"