import torch
import torch.nn as nn
import torch.optim as optim
from heat_exchanger_physics import heat_exchanger_residual


class rMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2):
        super(rMLP, self).__init__()
        
        # Layer 1
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.skip1 = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Layer 2
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)  # Will predict both temperatures

        # Xavier initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h1 = self.bn1(torch.relu(self.linear1(x))) + self.skip1(x)
        h2 = self.bn2(torch.relu(self.linear2(h1))) + self.skip2(h1)
        y = self.output(h2)
        return y


def empirical_loss(pred, target):
    return nn.MSELoss()(pred, target)

def physics_loss(x, model, pde_func):
    # Autograd requires inputs with gradients
    x.requires_grad_(True)
    y = model(x)

    # Define physics-based residual (user supplies PDE function)
    residual = pde_func(x, y)
    return torch.mean(residual**2)


def train_model(model, train_data, train_targets, pde_func, epochs=1000, 
                lr=1e-3, λ1=1.0, λ2=0.01, use_meta_learning=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialise weights for physics components as per paper (https://www.sciencedirect.com/science/article/pii/S0098135425002157?via%3Dihub)
    # Paper: w_i = 100 initially, λ2*w_i = 1
    if use_meta_learning:
        # Initialize weights as trainable parameters
        w = torch.nn.Parameter(torch.ones(1) * 100.0, requires_grad=True)
        # Learning rate for meta-learning (η in equation 21)
        meta_lr = 0.01
    else:
        # Fixed weights when not using meta-learning (PI-rMLP case)
        w = torch.tensor([100.0])
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(train_data)
        
        # Compute losses
        loss_emp = empirical_loss(pred, train_targets)
        loss_pde = physics_loss(train_data.clone(), model, pde_func)
        
        # Total loss with weighted physics component (λ2*w*loss_pde)
        loss = λ1 * loss_emp + λ2 * w * loss_pde
        
        # Backward pass for model parameters
        loss.backward(retain_graph=use_meta_learning)
        optimizer.step()
        
        # Meta-learning step: update the weights if enabled
        # Following equation (21): w_i = w_i - η * ∂L/∂w_i
        if use_meta_learning:
            # Get gradient of loss with respect to w
            grad_w = w.grad
            
            if grad_w is not None:
                # Update w directly using equation (21)
                with torch.no_grad():
                    w.data -= meta_lr * grad_w
                    # Projection step: w_i = max(1, w_i) as mentioned in the paper
                    w.data = torch.clamp(w.data, min=1.0)
                
                # Zero the accumulated gradients
                w.grad.zero_()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Total Loss={loss.item():.6f}, "
                  f"Empirical={loss_emp.item():.6f}, "
                  f"PDE={loss_pde.item():.6f}, "
                  f"w={w.item():.6f}")
            

def export_to_onnx(model, train_data):
    """Export the trained model to ONNX format using a real sample from training data."""
    # Ensure model is in evaluation mode
    model.eval()
    
    # Use first sample from training data
    example_input = train_data[0:1]  # First sample with batch dimension of 1
    
    # Export to ONNX
    torch.onnx.export(
        model,                           # The trained model
        example_input,                   # Example input from real data
        "models/heat_exchanger_pinn.onnx",  # Output file path
        export_params=True,              # Store trained parameters
        opset_version=12,                # ONNX version
        do_constant_folding=True,        # Optimization
        input_names=['input'],           # Name of model inputs
        output_names=['output'],         # Name of model outputs
        dynamic_axes={'input': {0: 'batch_size'},    # Variable dimensions
                      'output': {0: 'batch_size'}}
    )
    print("\nModel exported to ONNX format: models/heat_exchanger_pinn.onnx")
    

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    
    # Load the generated data from the data directory
    X_train = torch.load("data/heat_exchanger_inputs.pt")
    y_train = torch.load("data/heat_exchanger_outputs.pt")
    print(f"Loaded {len(X_train)} samples for training")
    
    # Create the model - make sure output_dim is 2 for both temperatures
    model = rMLP(input_dim=4, hidden_dim=32, output_dim=2)
    
    # Train with meta-learning physics-informed loss (MLPI-rMLP)
    train_model(
        model, 
        X_train, 
        y_train, 
        heat_exchanger_residual,
        epochs=2000,
        lr=3e-4,                 # Model learning rate
        λ1=1.0,                  # Fixed weight for empirical loss (fixed at 1 per paper)
        λ2=0.01,                 # Weight for physics loss (0.01 per paper)
        use_meta_learning=True   # Enable meta-learning for physics weights
    )
    
    # Test the model
    model.eval()  # Set to evaluation mode
    test_input = torch.tensor([[0.18, 0.15, 390.0, 295.0]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(test_input)
    
    print("\nTest prediction:")
    print(f"Input: m_hot={test_input[0,0]:.3f} kg/s, m_cold={test_input[0,1]:.3f} kg/s")
    print(f"       T_hot_in={test_input[0,2]:.1f}K, T_cold_in={test_input[0,3]:.1f}K")
    print(f"Predicted: T_hot_out={prediction[0,0]:.2f}K, T_cold_out={prediction[0,1]:.2f}K")
    
    # Save the trained model to the models directory
    torch.save(model.state_dict(), "models/heat_exchanger_pinn_model.pt")
    print("Model saved to models/heat_exchanger_pinn_model.pt")

    export_to_onnx(model, X_train)