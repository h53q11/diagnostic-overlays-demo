
# diagnostic-overlays-demo  
Minimal demonstrator for diagnostic overlay framework (toy models + safe diagnostic overlays)

---

## Overview

This repository provides a **simplified, conceptual demonstrator** accompanying the manuscript on *multimodal diagnostic overlays for tabular deep learning*.  
It is designed to illustrate the **structure** of diagnostic overlays in a lightweight, reproducible form **without exposing the full experimental pipeline** used in the study.

The demonstrator includes:

- A **toy MLP** trained on synthetic tabular regression data  
- A **toy GRU** trained on synthetic sequential data  
- Simple diagnostic overlays:
  - ℓ2‑norm of parameter gradients  
  - variance of hidden activations  
  - training loss  
- A safe **model‑initialisation example**  
- A safe **attribution demonstrator** (gradient attribution + integrated gradients)  
- A placeholder **perturbation function**  
- A unified configuration dictionary and reproducibility seed  

All components are intentionally minimal and conceptual.

---

## Purpose

The demonstrator supports reproducibility by showing:

- how diagnostic signals can be collected during optimisation  
- how overlays can be plotted over training steps  
- how attribution can be computed in principle  
- how model initialisation and configuration can be structured  

It is **not** intended to reproduce the full multimodal, behaviour‑aware diagnostic overlays developed in the main manuscript.  
Those overlays involve additional modules, temporal analysis, perturbation‑based metrics, and multimodal integration that cannot be released due to methodological complexity and ongoing research dependencies.

---

## What the demonstrator includes

- `run_demo.py`  
  - MLP diagnostic overlay  
  - GRU diagnostic overlay  
  - attribution demo (gradient + IG)  
  - model initialisation demo  
  - synthetic datasets  
  - reproducibility seed  
  - safe perturbation placeholder  
  - unified configuration dictionary  

- `requirements.txt`  
  Minimal dependencies required to run the demonstrator.

---

## What the demonstrator does **not** include

- full multimodal diagnostic overlay system  
- behaviour‑aware perturbation metrics  
- temporal instability analysis  
- multimodal integration pipeline  
- experimental configuration used in the manuscript  
- real datasets  
- proprietary modules or research code  

These components remain private and are not required for conceptual reproducibility.

---

## Running the demonstrator

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all demos:

```bash
python run_demo.py
```

This will execute:

- MLP overlay demo  
- GRU overlay demo  
- model initialisation demo  
- attribution demo  

Each produces diagnostic plots illustrating the conceptual workflow.

---

## Citation

If you use this demonstrator, please cite the accompanying manuscript.

---

