# diagnostic-overlays-demo
Minimal demonstrator for diagnostic overlay framework (toy model + overlay)
# Diagnostic Overlays – Toy Demonstrator 

This repository provides a minimal toy example accompanying the manuscript
on multimodal diagnostic overlays for tabular deep learning.

The demo trains a small MLP on synthetic tabular data and visualises a simple
gradient-noise overlay during training.

# Toy Demonstrator for Reproducibility

To support reproducibility requirements and provide a minimal illustration of the diagnostic overlay workflow, a simplified demonstrator is supplied in the accompanying repository. The demonstrator is intentionally independent of the full multimodal overlay system described in the main manuscript and is designed solely to show the structure of an overlay pipeline in a lightweight, reproducible form.

The toy example trains a small multilayer perceptron (MLP) on a synthetic tabular regression dataset generated using `make_regression`. During training, two simple diagnostic signals are recorded at each optimisation step: (i) the ℓ2‑norm of the parameter gradients, used as a proxy for gradient‑noise fluctuations, and (ii) the variance of hidden‑layer activations, used as a proxy for activation stability. These signals are collected alongside the batch loss and plotted as aligned traces over training steps, forming a basic “overlay” that illustrates how internal model behaviour can be monitored during optimisation.

The demonstrator consists of a single Python script (`run_demo.py`) and a minimal dependency file (`requirements.txt`). It is intended as a conceptual example only and does not reproduce the full multimodal, multicoloured diagnostic overlays developed for the main study. All code required to run the toy example is provided in the public repository referenced in the Code Availability Statement.


## Installation
pip install -r requirements.txt

## Usage
python run_demo.py
