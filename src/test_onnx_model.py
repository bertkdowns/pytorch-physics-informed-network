import torch
import onnxruntime as ort
import os
import sys
from residual_network import rMLP

def test_onnx_pytorch_consistency():
    """
    Test that the exported ONNX model produces the same results as the PyTorch model.
    """
    # Check that model files exist
    pt_path = "models/heat_exchanger_pinn_model.pt"
    onnx_path = "models/heat_exchanger_pinn.onnx"
    
    if not os.path.exists(pt_path) or not os.path.exists(onnx_path):
        print(f"ERROR: Model files not found. Please run training first with the script ./run_pinn.sh.")
        return False
    
    print("\n===== Testing ONNX Model Consistency =====")
    
    # 1. Load the PyTorch model
    print("Loading PyTorch model...")
    pytorch_model = rMLP(input_dim=4, hidden_dim=32, output_dim=2)
    pytorch_model.load_state_dict(torch.load(pt_path))
    pytorch_model.eval()
    
    # 2. Create a test input
    print("Creating test inputs...")
    test_inputs = [
        torch.tensor([[0.18, 0.15, 390.0, 295.0]], dtype=torch.float32),  # Standard case
        torch.tensor([[0.20, 0.10, 400.0, 300.0]], dtype=torch.float32),  # Different values
        torch.tensor([[0.18, 0.15, 390.0, 295.0], 
                     [0.20, 0.10, 400.0, 300.0]], dtype=torch.float32),  # Batch inference
    ]
    
    # 3. Initialise ONNX Runtime
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # 4. Run comparison tests
    all_passed = True
    for i, test_input in enumerate(test_inputs):
        print(f"\nTest case {i+1} (shape: {test_input.shape}):")
        
        # PyTorch prediction
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
        
        # ONNX prediction
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        onnx_output = torch.tensor(ort_output)
        
        # Compare results
        abs_diff = torch.abs(pytorch_output - onnx_output)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Print results
        print(f"  Maximum absolute difference: {max_diff:.8f}")
        print(f"  Average absolute difference: {mean_diff:.8f}")
        
        # Comparison for the first example in each batch
        print("  First sample comparison:")
        print(f"    PyTorch: T_hot_out={pytorch_output[0,0]:.4f}K, T_cold_out={pytorch_output[0,1]:.4f}K")
        print(f"    ONNX:    T_hot_out={onnx_output[0,0]:.4f}K, T_cold_out={onnx_output[0,1]:.4f}K")
        
        # Check if the difference is acceptable (using a small threshold)
        if max_diff > 1e-3:
            print("  [FAILED] Models produce different results!")
            all_passed = False
        else:
            print("  [PASSED] Models match!")
    
    return all_passed

if __name__ == "__main__":
    success = test_onnx_pytorch_consistency()
    # Exit with code 0 if successful, 1 if not
    sys.exit(0 if success else 1)