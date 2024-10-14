"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y


# - id
def id(x: float) -> float:
    """$f(x) = x$"""
    return x


# - add
def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y


# - neg
def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x


# - lt
def lt(x: float, y: float) -> float:
    """$f(x) = $ 1.0 if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0


# - eq
def eq(x: float, y: float) -> float:
    """$f(x) = $ 1.0 if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0


# - max
def max(x: float, y: float) -> float:
    """$f(x) = $ x if x is greater than y else y"""
    return x if x > y else y


# - is_close
def is_close(x: float, y: float) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    return (x - y < 1e-2) and (y - x < 1e-2)


# - sigmoid
def sigmoid(x: float) -> float:
    r"""$f(x) = \frac{1.0}{(1.0 + e^{-x})}$ if $x >= 0$ else $\frac{e^x}{(1.0 + e^{x})}$"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """$f(x) = $ x if x is greater than 0, else 0"""
    return x if x > 0 else 0.0


EPS = 1e-6


# - log
def log(x: float) -> float:
    """$f(x) = log(x)$"""
    return math.log(x + EPS)


# - exp
def exp(x: float) -> float:
    """$f(x) = e^{x}$"""
    return math.exp(x)


# - abs
def abs(x: float) -> float:
    """Returns the absolute value of a float."""
    return x if x >= 0 else -x


# - log_back
def log_back(x: float, d: float) -> float:
    r"""If $f = log$ as above, compute $d \times f'(x)$"""
    return d / (x + EPS)


# - inv
def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return 1.0 / x


# - inv_back
def inv_back(x: float, d: float) -> float:
    r"""If $f = inv$ as above, compute $d \times f'(x)$"""
    return -(1.0 / x**2) * d


# - relu_back
def relu_back(x: float, d: float) -> float:
    r"""If $f = relu$ as above, compute $d \times f'(x)$"""
    return d if x > 0 else 0.0


# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    Args:
    ----
    fn: Function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    return map(neg)(ls)


# - zipWith
def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    Args:
    ----
        fn: combine two values

    Returns:
    -------
    A function that takes two lists, applies `fn` to each pair of elements, and returns a new list

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


# Use these to implement
# - negList : negate a list


# - addLists : add two lists together
def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
        fn: Function from two values to one value.
        start: start value $x_0$

    Returns:
    -------
    A function that takes a list, applies `fn` to each element, and returns a new list
    """  # noqa: D413

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


# - sum: sum lists
def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    return reduce(add, 0.0)(ls)


# - prod: take the product of lists
def prod(ls: Iterable[float]) -> float:
    """Produce of a list using `reduce` and `mul`."""
    return reduce(mul, 1.0)(ls)
