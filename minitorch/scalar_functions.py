from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Run the backward function, which computes the derivative with respect to inputs.

        Args:
        ----
        ctx : Context
            The context containing saved values from the forward pass.
        d_out : float
            The derivative of the output.

        Returns:
        -------
        Tuple[float, ...]
            The derivatives with respect to each input.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Run the forward function, which computes the output of the function.

        Args:
        ----
        ctx : Context
            The context to store information for the backward pass.
        *inps : float
            The inputs to the function.

        Returns:
        -------
        float
            The output of the function.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to scalar values and track history for backpropagation.

        Args:
        ----
        *vals : ScalarLike
            Scalar-like values to apply the function on.

        Returns:
        -------
        Scalar
            A scalar representing the result with tracking for backpropagation.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):
    """Addition function."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The first input value.
        b : float
            The second input value.

        Returns:
        -------
        float
            The result of a + b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        Tuple[float, ...]
            The derivatives of the inputs, both equal to d_output.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Logarithm function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for logarithm.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The input value.

        Returns:
        -------
        float
            The result of the logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for logarithm.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        float
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The first input value.
        b : float
            The second input value.

        Returns:
        -------
        float
            The result of a * b.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        Tuple[float, float]
            The derivatives of the inputs.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The input value.

        Returns:
        -------
        float
            The result of 1/a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        float
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The input value.

        Returns:
        -------
        float
            The negation of a.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        float
            The derivative of the input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The input value.

        Returns:
        -------
        float
            The result of the sigmoid function on a.

        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        float
            The derivative of the input.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The input value.

        Returns:
        -------
        float
            The result of the ReLU function on a.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        float
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The input value.

        Returns:
        -------
        float
            The result of exp(a).

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        float
            The derivative of the input.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less-than function that returns 1.0 if a < b, otherwise 0.0."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less-than comparison.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The first input value.
        b : float
            The second input value.

        Returns:
        -------
        float
            1.0 if a < b, else 0.0.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less-than comparison.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        Tuple[float, float]
            The gradients for a and b (both 0.0).

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function that returns 1.0 if a == b, otherwise 0.0."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality comparison.

        Args:
        ----
        ctx : Context
            The context for the operation.
        a : float
            The first input value.
        b : float
            The second input value.

        Returns:
        -------
        float
            1.0 if a == b, else 0.0.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality comparison.

        Args:
        ----
        ctx : Context
            The context containing saved values.
        d_output : float
            The derivative of the output.

        Returns:
        -------
        Tuple[float, float]
            The gradients for a and b (both 0.0).

        """
        return 0.0, 0.0
