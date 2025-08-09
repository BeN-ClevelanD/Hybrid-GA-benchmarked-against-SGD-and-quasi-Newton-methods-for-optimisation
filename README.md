# Comparative Study of Optimisation Algorithms for Feed-Forward Neural Networks

This repository contains a reproducible comparative study of four optimisation strategies applied to a regression-task feed-forward neural network (FFN) implemented in PyTorch.

## Overview

The project compares:

1. **Stochastic Gradient Descent (SGD)** — hyperparameters tuned via Optuna.
2. **L-BFGS** — full-batch quasi-Newton optimisation, tuned via Optuna.
3. **Genetic Algorithm (GA)** — BLX-α crossover, Gaussian mutation.
4. **Hybrid GA→SGD** — GA with periodic SGD refinement.

The network is a fixed architecture (two hidden layers, 24 ReLU units each) with Xavier parameter initialisation.

## Repository Structure

- `project_code.ipynb` — primary notebook containing all experiments and results.
- `project_code.html` — HTML export of the notebook.

## Data & Preprocessing

The scripts expect preprocessed training and validation tensors:
```python
X_train, y_train, X_val, y_val  # torch.Tensors, scaled via StandardScaler

Dependencies
Python 3.10+
PyTorch
NumPy
pandas
scikit-learn
matplotlib
Optuna
