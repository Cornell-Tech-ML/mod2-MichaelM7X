"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Type

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_functions import (
    Copy,
    Inv,
    MatMul,
    Mul,
    Neg,
    Add,
    Sigmoid,
    ReLU,
    Exp,
    Log,
    LT,
    EQ,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union
    import numpy.typing as npt
    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None  # Changed from Function to Type[Function]
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        self._is_boolean = False

        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, requires_grad: bool = True) -> None:
        """Sets whether this tensor requires gradient computation."""
        if requires_grad:
            self.history = History()
        else:
            self.history = None

    def requires_grad(self) -> bool:
        """Checks if this tensor requires gradient computation."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Convert the tensor to a NumPy array."""
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Ensure that the input is a tensor, converting if necessary."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float."""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data."""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data."""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Return a tensor filled with zeros."""

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backpropagation graph."""
        return Tensor(self._tensor, backend=self.backend)

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate derivative during backpropagation."""
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Check if this is a leaf node in the computation graph."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if this tensor is constant (has no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables in the computation graph."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for backpropagation."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, self._ensure_tensor(d_in).expand(inp.shape))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Compute the backward pass for this tensor."""
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    # Tensor operations
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __neg__(self) -> "Tensor":
        """Negation of the tensor."""
        return Neg.apply(self)

    def log(self) -> "Tensor":
        """Apply logarithm to this tensor."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Apply exponentiation to this tensor."""
        return Exp.apply(self)

    def sigmoid(self) -> Tensor:
        """Apply sigmoid function to this tensor."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Apply ReLU (Rectified Linear Unit) function to this tensor."""
        return ReLU.apply(self)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        return EQ.apply(self, self._ensure_tensor(b))

    @property
    def shape(self) -> UserShape:
        """Get the shape of the tensor."""
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Get the total number of elements in the tensor."""
        return int(operators.prod(self.shape))

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sum over all elements in the tensor, or along a dimension."""
        if dim is None:
            return Tensor.make(
                [float(np.sum(self._tensor._storage))], (1,), backend=self.backend
            )
        else:
            return self.f.add_reduce(self, dim)

    def zeros_like(self) -> Tensor:
        """Create a tensor filled with zeros having the same shape as this tensor."""
        return Tensor.make([0.0] * self.size, self.shape, backend=self.backend)

    def permute(self, *order: int) -> Tensor:
        """Permute the dimensions of this tensor."""
        new_strides = [self._tensor.strides[i] for i in order]
        new_shape = [self.shape[i] for i in order]
        return Tensor(
            TensorData(self._tensor._storage, new_shape, new_strides),
            backend=self.backend,
        )

    def __radd__(self, b: TensorLike) -> Tensor:
        """Right addition."""
        return self.__add__(b)

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Right multiplication."""
        return self.__mul__(b)

    def __rsub__(self, b: TensorLike) -> Tensor:
        """Right subtraction."""
        return self._ensure_tensor(b).__sub__(self)

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Right division."""
        return self._ensure_tensor(b).__truediv__(self)

    def view(self, *shape: int) -> Tensor:
        """Reshape this tensor to the specified shape."""
        new_size = int(operators.prod(shape))
        assert (
            new_size == self.size
        ), "New shape must have the same number of elements as the original."
        return Tensor(TensorData(self._tensor._storage, shape), backend=self.backend)

    def expand(self, shape: UserShape) -> Tensor:
        """Broadcast this tensor to a new shape."""
        # Implement broadcasting directly here
        new_data = self._tensor.broadcast_to(shape)
        return Tensor(new_data, backend=self.backend)

    def all(self) -> Tensor:
        """Return a Tensor of shape (1,) with the logical AND over all elements."""
        result = float(np.all(self.to_numpy()))
        return Tensor.make([result], (1,), backend=self.backend)

    def is_close(self, other: Tensor, atol: float = 1e-8) -> Tensor:
        """Check if this tensor is element-wise close to another tensor."""
        result = (self - other).abs() < atol
        result._is_boolean = True  # Custom attribute to indicate it's a boolean tensor
        return result

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean over all elements or along a dimension."""
        total = self.sum(dim)
        count = self.size if dim is None else self.shape[dim]
        return total / count

    def __bool__(self) -> bool:
        """Allow tensor to be used in boolean contexts if it contains a single boolean value."""
        if self.size == 1 and hasattr(self, "_is_boolean") and self._is_boolean:
            return bool(self.item())
        raise ValueError(
            "The truth value of a Tensor with more than one value is ambiguous"
        )

    def elementwise_is_close(self, other: Tensor, atol: float = 1e-8) -> Tensor:
        """Element-wise comparison of tensors within a tolerance."""
        return (self - other).abs() < atol

    def abs(self) -> Tensor:
        """Compute the absolute value element-wise."""
        return self.f.abs_map(self)

    def zero_grad_(self) -> None:
        """Set the gradient of this tensor to zero."""
        if self.grad is not None:
            self.grad._tensor._storage.fill(0.0)


# End of the Tensor class
