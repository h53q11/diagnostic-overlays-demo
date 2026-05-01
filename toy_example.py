import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_toy_data(n_samples=2000, n_features=10, noise=5.0, random_state=0):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)
    return X, y


def compute_grad_norm(model):
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                total += (p.grad.detach() ** 2).sum().item()
    return np.sqrt(total)


def main():
    # 1. Data
    X, y = make_toy_data()
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Model, loss, optimiser
    model = MLP(in_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training with gradient-noise overlay
    grad_norms = []
    losses = []

    n_epochs = 10
    step = 0
    for epoch in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()

            # record gradient norm before the optimiser step
            grad_norm = compute_grad_norm(model)
            grad_norms.append((step, grad_norm))
            losses.append((step, loss.item()))

            optimizer.step()
            step += 1

    # 4. Plot overlay: gradient norm over training steps
    steps, gvals = zip(*grad_norms)
    _, lvals = zip(*losses)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    color1 = "tab:blue"
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Gradient norm", color=color1)
    ax1.plot(steps, gvals, color=color1, label="Gradient norm")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Loss", color=color2)
    ax2.plot(steps, lvals, color=color2, alpha=0.6, label="Loss")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.tight_layout()
    plt.title("Toy MLP: Gradient-Noise Overlay During Training")
    plt.show()


if __name__ == "__main__":
    main()
