from .manifold_mixins import LinBoundedFilterMixin
from .abstract_particle_filter import AbstractParticleFilter


class LinBoundedParticleFilter(AbstractParticleFilter, LinBoundedFilterMixin):
    pass
