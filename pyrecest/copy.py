"""Compatibility copy helpers exposed at the package level."""

from copy import deepcopy as _deepcopy

from pyrecest.backend import copy

__all__ = ["copy", "deepcopy"]


def deepcopy(value, memo=None):
    """Deep-copy a Python object."""
    return _deepcopy(value, memo)
