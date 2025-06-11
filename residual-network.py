import torch
import torch.nn as nn
import torch.optim as optim


class rMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=4, output_dim=1):
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
        self.output = nn.Linear(hidden_dim, output_dim)

        # Xavier initialization
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


def train_model(model, train_data, train_targets, pde_func, epochs=1000, lr=1e-3, λ1=1.0, λ2=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        pred = model(train_data)

        # Compute losses
        loss_emp = empirical_loss(pred, train_targets)
        loss_pde = physics_loss(train_data.clone(), model, pde_func)

        loss = λ1 * loss_emp + λ2 * loss_pde

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Total Loss={loss.item():.6f}, Empirical={loss_emp.item():.6f}, PDE={loss_pde.item():.6f}")


# Example placeholder PDE function
def example_pde(x, y):
    """Dummy PDE: dy/dx0 + dy/dx1 = 0"""
    grads = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    return grads[:, 0] + grads[:, 1]  # sum of partial derivatives w.r.t. x0 and x1


# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)

    # Generate dummy training data
    X_train = torch.rand((64, 4))  # 64 samples, 4 features: [m˙WF, m˙B, TinWF, TinB]
    y_train = torch.rand((64, 1))  # target output

    model = rMLP()
    train_model(model, X_train, y_train, example_pde, epochs=1000, lr=1e-3, λ1=1.0, λ2=0.1)
