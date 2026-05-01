import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


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
# 2. Toy dataset
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
# 4. Main demo
# ---------------------------------------------------------
def main():
    X, y = make_toy_data()
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MLP(in_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    overlay_grad = []
    overlay_actvar = []
    overlay_loss = []

    step = 0
    for epoch in range(5):
        for xb, yb in loader:
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

    # -----------------------------------------------------
    # 5. Plot overlays
    # -----------------------------------------------------
    steps, gvals = zip(*overlay_grad)
    _, avars = zip(*overlay_actvar)
    _, lvals = zip(*overlay_loss)

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(steps, gvals, color="blue")
    ax[0].set_ylabel("Gradient Norm")
    ax[0].set_title("Diagnostic Overlay: Gradient Noise")

    ax[1].plot(steps, avars, color="green")
    ax[1].set_ylabel("Activation Variance")
    ax[1].set_title("Diagnostic Overlay: Activation Stability")

    ax[2].plot(steps, lvals, color="orange")
    ax[2].set_ylabel("Loss")
    ax[2].set_xlabel("Training Step")
    ax[2].set_title("Training Loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
