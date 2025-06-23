#!/bin/bash
# Test ONNX model consistency with PyTorch model

set -e  # Exit on error

echo "===== ONNX Model Consistency Test ====="

# Make sure directories exist
mkdir -p data models

# Check if models exist, if not run training first
if [ ! -f "models/heat_exchanger_pinn_model.pt" ] || [ ! -f "models/heat_exchanger_pinn.onnx" ]; then
    echo "[1/2] Models not found. Running training first..."
    ./run_pinn.sh
else
    echo "[1/2] Models found. Skipping training."
fi

# Run the ONNX test
echo "[2/2] Testing ONNX model consistency..."
python src/test_onnx_model.py
TEST_RESULT=$?

# Check result
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Test passed successfully! ONNX model is consistent with PyTorch model."
else
    echo "❌ Test failed! ONNX model differs from PyTorch model."
    exit 1
fi

echo "===== Test Complete ====="