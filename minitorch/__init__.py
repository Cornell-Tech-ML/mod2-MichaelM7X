"""minitorch: A minimal neural network framework with autodifferentiation support.

This package contains various modules for building and training neural networks.
It includes tensor operations, autodifferentiation, optimizers, datasets, and testing utilities.

Modules:
--------
- tensor_data: Provides utility functions and data structures for handling tensor operations.
- tensor: Defines the core Tensor class with operations and autodifferentiation support.
- tensor_ops: Contains operations for manipulating tensors.
- tensor_functions: Implements common mathematical functions for tensors.
- autodiff: Handles the autodifferentiation and backpropagation logic.
- scalar: Provides scalar operations for basic arithmetic.
- scalar_functions: Implements scalar mathematical functions.
- optim: Contains optimizers for training neural networks.
- datasets: Includes utility functions for generating and managing datasets.
- testing: Provides testing utilities for mathematical operations and variables.
- module: Defines the core module class for building neural network layers and models.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .tensor import Tensor


def is_close(a: Tensor, b: Tensor, atol: float = 1e-8) -> bool:
    """Check if two tensors are element-wise close within a tolerance."""
    return a.is_close(b, atol=atol)
