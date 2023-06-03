import numpy as np
from beartype import beartype

from .hypertoroidal_particle_filter import HypertoroidalParticleFilter


class ToroidalParticleFilter(HypertoroidalParticleFilter):
    @beartype
    def __init__(self, n_particles: int | np.int32 | np.int64):
        HypertoroidalParticleFilter.__init__(self, n_particles, 2)
