from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


# ## Task 1.1
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
    # TODO: Implement for Task 1.1.
    vals_list = list(vals)
    vals_list[arg] += epsilon
    f_plus = f(*vals_list)

    # Compute f(vals[arg] - epsilon)
    vals_list[arg] -= 2 * epsilon
    f_minus = f(*vals_list)

    # Return the central difference approximation of the derivative
    return (f_plus - f_minus) / (2 * epsilon)
    raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...  # noqa: D102

    @property
    def unique_id(self) -> int: ...  # noqa: D102

    def is_leaf(self) -> bool: ...  # noqa: D102

    def is_constant(self) -> bool: ...  # noqa: D102

    @property
    def parents(self) -> Iterable["Variable"]: ...  # noqa: D102

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...  # noqa: D102


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
    variable: The right-most variable

    Returns:
    -------
    Non-constant Variables in topological order starting from the right.

    """
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
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # Dictionary to store derivatives for each variable
    derivatives_dict = {variable.unique_id: deriv}

    top_sort = topological_sort(variable)

    # Iterate through the topological order and calculate the derivatives
    for curr_var in top_sort:
        if curr_var.is_leaf():
            continue

        # Get the derivatives of the current variable
        var_n_der = curr_var.chain_rule(derivatives_dict[curr_var.unique_id])

        # Accumulate the derivative for each parent of the current variable
        for var, deriv in var_n_der:
            if var.is_leaf():
                var.accumulate_derivative(deriv)
            else:
                if var.unique_id not in derivatives_dict:
                    derivatives_dict[var.unique_id] = deriv
                else:
                    derivatives_dict[var.unique_id] += deriv
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:  # noqa: D102
        return self.saved_values
