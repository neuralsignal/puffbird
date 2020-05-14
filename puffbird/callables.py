"""
Iterables
=========

Various iterable functions that can be used instead of `iter`
to pass to the `puffy_to_long` or `to_long` method in `FrameEngine`.
"""

from typing import Callable


class CallableContainer:
    """
    Container of callables, that accept one argument.

    Parameters
    ----------
    default_callable : callable
        Default callable used when argument passed does not correspond
        to an assigned instance.

    Notes
    -----
    Each callable can be assigned to a specific instances using the
    :obj:`~CallableContainer.add` method. This method accepts a callable
    and a class or a tuple of class objects. When the container is called
    it will check if the single argument passed corresponds to an instance of
    the class objects and if so use the assigned callable with the argument
    passed.
    """

    def __init__(self, default_callable: Callable):
        self._default_callable = default_callable
        self._callables = []

    def add(self, a_callable: Callable, classes):
        """
        Add a new callable with allowed classes.
        """
        self._callables.append(
            (a_callable, classes)
        )
        return self

    def __call__(self, x):
        """
        Check type of `x` and then use the appropriate callable.
        """
        for callme, classes in self._callables:
            if isinstance(x, classes):
                return callme(x)
        return self._default_callable(x)

    def __repr__(self):
        return f"{type(self).__name__}({self._default_callable.__name__})"
