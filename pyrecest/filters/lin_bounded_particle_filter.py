from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import LinBoundedFilterMixin


class LinBoundedParticleFilter(AbstractParticleFilter, LinBoundedFilterMixin):
    pass
