"""Numpy based random backend."""

import numpy as _np
from numpy.random import default_rng as _default_rng
from numpy.random import randint, seed, multinomial

from .._shared_numpy.random import choice, multivariate_normal, normal, rand, uniform