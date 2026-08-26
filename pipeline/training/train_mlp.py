
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from pipeline.models.models import MLP
from pipeline.data.toy_data import make_toy_data
from pipeline.diagnostics.basic_metrics import gradient_norm, activation_variance


# ---------------------------------------------------------
#  MLP demo
# ---------------------------------------------------------
def _run_mlp_demo(config):
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

def run_mlp_demo():
    config = {
        "n_samples": 2000,
        "n_features": 10,
        "noise": 5.0,
        "batch_size": 64,
        "epochs": 5,
        "mlp_hidden_dim": 32,
        "learning_rate": 1e-3,
    }
    return _run_mlp_demo(config)

def run_mlp_demo():
    config = {
        "n_samples": 2000,
        "n_features": 10,
        "noise": 5.0,
        "batch_size": 64,
        "epochs": 5,
        "mlp_hidden_dim": 32,
        "learning_rate": 1e-3,
    }
    return _run_mlp_demo(config)
