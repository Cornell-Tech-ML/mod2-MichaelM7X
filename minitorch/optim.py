from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    """Base class for optimizers in the minitorch framework.

    Attributes
    ----------
    parameters : Sequence[Parameter]
        The parameters to be optimized.

    """

    def __init__(self, parameters: Sequence[Parameter]):
        """Initialize the optimizer with the given parameters.

        Args:
        ----
        parameters : Sequence[Parameter]
            The parameters to be optimized.

        """
        self.parameters = parameters


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    Attributes
    ----------
    lr : float
        Learning rate for the optimizer.

    """

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Initialize the SGD optimizer.

        Args:
        ----
        parameters : Sequence[Parameter]
            The parameters to be optimized.
        lr : float, optional
            Learning rate for the optimizer (default is 1.0).

        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Set the gradients of all parameters to zero.
        This method should be called before each backward pass to prevent accumulation of gradients.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Update the parameters based on their gradients and the learning rate.
        This method performs the gradient descent step.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
