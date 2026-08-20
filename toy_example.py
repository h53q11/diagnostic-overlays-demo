import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# NOTE:
# This demonstrator illustrates basic model structures (MLP and GRU),
# synthetic datasets, and simple diagnostics (gradient norm, activation variance, loss).
# It does NOT include the full experimental pipeline used in the manuscript.


# ---------------------------------------------------------
# 0. Reproducibility helpers
# ---------------------------------------------------------
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cpu")


def apply_small_noise(x, epsilon=1e-3):
    """Safe placeholder for input perturbation."""
    return x + epsilon * torch.randn_like(x)


# ---------------------------------------------------------
# 1. Simple MLP model
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out, h2  # return activations for overlay


# ---------------------------------------------------------
# 2. Toy dataset (tabular)
# ---------------------------------------------------------
def make_toy_data(n_samples=2000, n_features=10, noise=5.0, random_state=0):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


# ---------------------------------------------------------
# 3. Overlay functions
# ---------------------------------------------------------
def gradient_norm(model):
    """Compute L2 norm of all gradients."""
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                total += (p.grad.detach() ** 2).sum().item()
    return np.sqrt(total)


def activation_variance(activations):
    """Simple diagnostic: variance of hidden activations."""
    return activations.var().item()


# ---------------------------------------------------------
# 4. MLP demo
# ---------------------------------------------------------
def run_mlp_demo(config):
    X, y = make_toy_data(
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        noise=config["noise"],
        random_state=0,
    )
    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = MLP(in_dim=X.shape[1], hidden_dim=config["mlp_hidden_dim"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    overlay_grad = []
    overlay_actvar = []
    overlay_loss = []

    step = 0
    for epoch in range(config["epochs"]):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds, acts = model(xb)
            loss = criterion(preds, yb)
            loss.backward()

            # --- Overlay signals ---
            gnorm = gradient_norm(model)
            avar = activation_variance(acts)

            overlay_grad.append((step, gnorm))
            overlay_actvar.append((step, avar))
            overlay_loss.append((step, loss.item()))

            optimizer.step()
            step += 1

        print(f"[MLP] Epoch {epoch+1}: last batch loss = {loss.item():.4f}")

    # Plot overlays
    steps, gvals = zip(*overlay_grad)
    _, avars = zip(*overlay_actvar)
    _, lvals = zip(*overlay_loss)

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(steps, gvals, color="blue")
    ax[0].set_ylabel("Gradient Norm")
    ax[0].set_title("MLP Diagnostic Overlay: Gradient Noise")

    ax[1].plot(steps, avars, color="green")
    ax[1].set_ylabel("Activation Variance")
    ax[1].set_title("MLP Diagnostic Overlay: Activation Stability")

    ax[2].plot(steps, lvals, color="orange")
    ax[2].set_ylabel("Loss")
    ax[2].set_xlabel("Training Step")
    ax[2].set_title("MLP Training Loss")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 5. Simple GRU model
# ---------------------------------------------------------
class SimpleGRU(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # final timestep
        return self.fc(last), out  # return output + hidden sequence


# ---------------------------------------------------------
# 6. Toy sequence dataset
# ---------------------------------------------------------
def make_sequence_data(n_samples=2000, seq_len=15, n_features=10):
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = X.mean(axis=(1, 2)).reshape(-1, 1).astype(np.float32)  # trivial target
    return X, y


def gru_activation_variance(hidden_seq):
    """Variance across all hidden states."""
    return hidden_seq.var().item()


# ---------------------------------------------------------
# 7. GRU demo
# ---------------------------------------------------------
def run_gru_demo(config):
    X, y = make_sequence_data(
        n_samples=config["n_samples"],
        seq_len=config["seq_len"],
        n_features=config["n_features"],
    )
    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = SimpleGRU(input_dim=X.shape[2], hidden_dim=config["gru_hidden_dim"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    overlay_loss = []
    overlay_actvar = []
    overlay_grad = []

    step = 0
    for epoch in range(config["epochs"]):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds, hidden_seq = model(xb)
            loss = criterion(preds, yb)
            loss.backward()

            # Diagnostics
            gnorm = gradient_norm(model)
            avar = gru_activation_variance(hidden_seq)

            overlay_loss.append((step, loss.item()))
            overlay_grad.append((step, gnorm))
            overlay_actvar.append((step, avar))

            optimizer.step()
            step += 1

        print(f"[GRU] Epoch {epoch+1}: last batch loss = {loss.item():.4f}")

    # Plot
    steps, gvals = zip(*overlay_grad)
    _, avars = zip(*overlay_actvar)
    _, lvals = zip(*overlay_loss)

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(steps, gvals, color="purple")
    ax[0].set_ylabel("Gradient Norm")
    ax[0].set_title("GRU Diagnostic: Gradient Noise")

    ax[1].plot(steps, avars, color="red")
    ax[1].set_ylabel("Activation Variance")
    ax[1].set_title("GRU Diagnostic: Hidden Stability")

    ax[2].plot(steps, lvals, color="black")
    ax[2].set_ylabel("Loss")
    ax[2].set_xlabel("Training Step")
    ax[2].set_title("GRU Training Loss")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 8. Demonstrator: model initialisation (safe version)
# ---------------------------------------------------------
class DemoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), output_dim=1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_initialisation_demo(X_tensor, y_tensor):
    """
    This function demonstrates a typical model initialisation workflow.
    It does NOT replicate the full experimental pipeline.
    """
    input_dim = X_tensor.shape[1]
    model = DemoMLP(input_dim=input_dim).to(device)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    target_mean = float(y_tensor.mean())
    target_std = float(y_tensor.std())

    print("\n[Initialisation Demo]")
    print("  Input dim:", input_dim)
    print("  Target mean:", target_mean)
    print("  Target std:", target_std)
    print("  Learning rate:", learning_rate)

    return model, optimizer, criterion

# ---------------------------------------------------------
# 9. Simple Attribution Demonstrator (safe)
# ---------------------------------------------------------

def simple_gradient_attribution(model, xb):
    """
    Safe placeholder for gradient-based attribution.
    Computes |d(output)/d(input)| for each feature.
    """
    xb = xb.clone().detach().requires_grad_(True)
    preds, _ = model(xb)
    pred = preds.mean()  # scalar
    pred.backward()

    return xb.grad.abs().mean(dim=0).detach()  # feature-level attribution


def simple_integrated_gradients(model, xb, steps=20):
    """
    Safe placeholder for Integrated Gradients.
    Uses a straight-line path from baseline to input.
    """
    baseline = torch.zeros_like(xb)
    scaled_inputs = [
        baseline + (float(i) / steps) * (xb - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    for x_scaled in scaled_inputs:
        x_scaled = x_scaled.clone().detach().requires_grad_(True)
        preds, _ = model(x_scaled)
        pred = preds.mean()
        pred.backward()
        grads.append(x_scaled.grad.detach())

    avg_grad = torch.stack(grads).mean(dim=0)
    return (xb - baseline) * avg_grad  # IG formula


def run_attribution_demo():
    """
    Demonstrates simple gradient and IG attribution.
    Does NOT replicate the real diagnostic overlay pipeline.
    """
    print("\nRunning Attribution Demo...")

    # Toy data
    X, y = make_toy_data(n_samples=128, n_features=10)
    X_tensor = torch.from_numpy(X).to(device)

    model = MLP(in_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train briefly
    for _ in range(3):
        preds, _ = model(X_tensor)
        loss = criterion(preds, torch.from_numpy(y).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Attribution on a single batch
    xb = X_tensor[:32]

    grad_attr = simple_gradient_attribution(model, xb)
    ig_attr = simple_integrated_gradients(model, xb)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].bar(range(len(grad_attr)), grad_attr.cpu().numpy())
    ax[0].set_title("Gradient Attribution")
    ax[0].set_xlabel("Feature")
    ax[0].set_ylabel("Importance")

    ax[1].bar(range(len(ig_attr.mean(dim=0))), ig_attr.mean(dim=0).cpu().numpy())
    ax[1].set_title("Integrated Gradients")
    ax[1].set_xlabel("Feature")
    ax[1].set_ylabel("Importance")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 10. Run all demos
# ---------------------------------------------------------
def run_all_demos():
    config = {
        "n_samples": 2000,
        "n_features": 10,
        "noise": 5.0,
        "batch_size": 64,
        "epochs": 5,
        "mlp_hidden_dim": 32,
        "gru_hidden_dim": 32,
        "learning_rate": 1e-3,
        "seq_len": 15,
    }

    print("Running MLP demo...")
    run_mlp_demo(config)

    print("\nRunning GRU demo...")
    run_gru_demo(config)

    print("\nRunning initialisation demo...")
    X, y = make_toy_data(
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        noise=config["noise"],
        random_state=0,
    )
    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    run_initialisation_demo(X_tensor, y_tensor)

    print("\nRunning attribution demo...")
    run_attribution_demo()


if __name__ == "__main__":
    set_seed(0)
    run_all_demos()

