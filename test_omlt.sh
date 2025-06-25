#!/bin/bash
# Test OMLT integration with ONNX model

set -e  # Exit on error

echo "===== Testing OMLT Integration ====="

# Check if models exist, if not run training first
if [ ! -f "models/heat_exchanger_pinn_model.pt" ] || [ ! -f "models/heat_exchanger_pinn.onnx" ]; then
    echo "[1/2] Models not found. Running training first..."
    ./run_pinn.sh
else
    echo "[1/2] Models found. Skipping training."
fi

echo "[2/2] Running OMLT test..."
python src/test_omlt.py
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Test passed successfully!"
else
    echo "❌ Test encountered a runtime error."
    exit 1
fi

echo "===== Test Complete ====="