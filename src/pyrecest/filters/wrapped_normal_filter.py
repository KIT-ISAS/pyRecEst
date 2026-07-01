import math
from collections.abc import Callable

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array, log, max, maximum, min, minimum, mod, pi
from pyrecest.distributions import CircularDiracDistribution, WrappedNormalDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin

_PROGRESSIVE_TAU_MESSAGE = "tau must be a positive finite scalar"
