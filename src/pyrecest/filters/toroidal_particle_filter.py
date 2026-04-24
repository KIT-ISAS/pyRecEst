from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import int32, int64

from .hypertoroidal_particle_filter import HypertoroidalParticleFilter
from .manifold_mixins import ToroidalFilterMixin


class ToroidalParticleFilter(HypertoroidalParticleFilter, ToroidalFilterMixin):
    def __init__(self, n_particles: Union[int, int32, int64]):
        ToroidalFilterMixin.__init__(self)
        HypertoroidalParticleFilter.__init__(self, n_particles, 2)
