
import torch
import matplotlib.pyplot as plt

from pipeline.models.models import DemoMLP


def simple_gradient_attribution(model, x):
    x = x.clone().requires_grad_(True)
    out = model(x)
    out.mean().backward()
    return x.grad.detach()


def simple_integrated_gradients(model, x, steps=20):
    baseline = torch.zeros_like(x)
    scaled_inputs = [(baseline + (i / steps) * (x - baseline)).clone().requires_grad_(True)
                     for i in range(steps + 1)]
    grads = []
    for s in scaled_inputs:
        out = model(s)
        out.mean().backward()
        grads.append(s.grad.detach())
    avg_grad = torch.stack(grads).mean(dim=0)
    return (x - baseline) * avg_grad

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

