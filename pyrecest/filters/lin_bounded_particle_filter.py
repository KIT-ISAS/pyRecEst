from .abstract_lin_bounded_filter import AbstractLinBoundedFilter
from .abstract_particle_filter import AbstractParticleFilter


class LinBoundedParticleFilter(AbstractParticleFilter, AbstractLinBoundedFilter):
    pass
