"""Differentiable TauREx forward models and optimizers.

The package exposes a JAX re-implementation of the transmission forward model
that keeps the taurex object graph as its source of truth, plus a retrieval
optimizer that uses the gradient of that model to fit far more quickly than the
sampling based optimizers.
"""

import jax


# The numpy model this is compared against works in double precision, so the
# port has to as well: the agreement between the two is only meaningful if the
# arithmetic is done to the same width. JAX defaults to single precision, so
# 64 bit mode is switched on here, before any of the arrays below are built.
jax.config.update("jax_enable_x64", True)

from .model import Atmosphere  # noqa: E402
from .optimizer import LaplaceOptimizer, PriorTransform  # noqa: E402


__all__ = ["Atmosphere", "LaplaceOptimizer", "PriorTransform"]
