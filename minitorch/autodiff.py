from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
    f : arbitrary function from n-scalar args to one value
    *vals : n-float values $x_0 \ldots x_{n-1}$
    arg : the number $i$ of the arg to compute the derivative
    epsilon : a small constant

    Returns:
    -------
    An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    ## ASSIGN1.1
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)
    ## END ASSIGN1.1


variable_count = 1


class Variable(Protocol):
    """Protocol for differentiable variables in the computation graph."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique ID for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is constant."""
        ...

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for the variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
    variable: The right-most variable in the graph.

    Returns:
    -------
    Non-constant variables in topological order starting from the right.

    """
    # ASSIGN1.4
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order
    # END ASSIGN1.4


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leaf nodes.

    Args:
    ----
    variable: The right-most variable.
    deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
    None. The result is written to the derivative values of each leaf through `accumulate_derivative`.

    """
    # ASSIGN1.4
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d
    # END ASSIGN1.4


@dataclass
class Context:
    """Context class used by `Function` to store information during the forward pass.

    Attributes
    ----------
    no_grad: Flag to indicate if gradients are being tracked.
    saved_values: Values saved during the forward pass to be used in the backward pass.

    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation.

        Args:
        ----
        values: The values to store for later use.

        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors from the forward pass."""
        return self.saved_values
