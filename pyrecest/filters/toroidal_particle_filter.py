from pyrecest.backend import int64
from pyrecest.backend import int32
import numpy as np
from beartype import beartype

from .hypertoroidal_particle_filter import HypertoroidalParticleFilter


class ToroidalParticleFilter(HypertoroidalParticleFilter):
    @beartype
    def __init__(self, n_particles: int | int32 | int64):
        HypertoroidalParticleFilter.__init__(self, n_particles, 2)
