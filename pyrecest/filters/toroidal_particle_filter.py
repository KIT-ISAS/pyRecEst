from typing import Union

from pyrecest.backend import int32, int64

from .hypertoroidal_particle_filter import HypertoroidalParticleFilter


class ToroidalParticleFilter(HypertoroidalParticleFilter):
    def __init__(self, n_particles: Union[int, int32, int64]):
        HypertoroidalParticleFilter.__init__(self, n_particles, 2)
