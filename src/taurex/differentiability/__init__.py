"""Differentiable TauREx forward models and optimizers.

The package exposes a PyTorch re-implementation of the transmission forward
model that keeps the taurex object graph as its source of truth, plus a
retrieval optimizer that uses the gradient of that model to fit far more
quickly than the sampling based optimizers.
"""

from .model import Atmosphere
from .optimizer import LaplaceOptimizer, PriorTransform


__all__ = ["Atmosphere", "LaplaceOptimizer", "PriorTransform"]
