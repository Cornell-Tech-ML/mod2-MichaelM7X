"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1

# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

from typing import Callable, Iterable, List


def mul(x: float, y: float) -> float:
    """:math:`f(x, y) = x * y`"""
    return x * y


def id(x: float) -> float:
    """:math:`f(x) = x`"""
    return x


def add(x: float, y: float) -> float:
    """:math:`f(x, y) = x + y`"""
    return x + y


def neg(x: float) -> float:
    """:math:`f(x) = -x`"""
    return -x


def lt(x: float, y: float) -> float:
    """:math:`f(x, y) = x < y`"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """:math:`f(x, y) = x == y`"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """:math:`f(x) =` x if x is greater than y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """:math:`f(x, y) = |x - y| < 1e-2`"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r""":math:`f(x) = \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else `\frac{e^x}{(1.0 + e^{x})}`"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """:math:`f(x) = max(0, x)`"""
    return max(0.0, x)


def log(x: float) -> float:
    """:math:`f(x) = log(x)`"""
    return math.log(x)


def exp(x: float) -> float:
    """:math:`f(x) = e^x`"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r""":math:`f(x) = \frac{d}{dx} log(x) = \frac{1}{x}`"""
    return d / x


def inv(x: float) -> float:
    r""":math:`f(x) = \frac{1}{x}`"""
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r""":math:`f(x) = \frac{d}{dx} \frac{1}{x} = -\frac{1}{x^2}`"""
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """:math:`f(x) =` d if x is greater than 0 else 0"""
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(func: Callable[[float], float], iter: Iterable[float]) -> List[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    return [func(x) for x in iter]


def zipWith(
    func: Callable[[float, float], float],
    iter1: Iterable[float],
    iter2: Iterable[float],
) -> List[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    return [func(x, y) for x, y in zip(iter1, iter2)]


def reduce(
    func: Callable[[float, float], float], iter: Iterable[float], ans: float
) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    for x in iter:
        ans = func(ans, x)
    return ans


def negList(iter: Iterable[float]) -> List[float]:
    """Negate all elements in a list using map"""
    return map(neg, iter)


def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> List[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add, iter1, iter2)


def sum(iter: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, iter, 0)


def prod(iter: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, iter, 1)
