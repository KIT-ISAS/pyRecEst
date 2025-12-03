from .lin_bounded_particle_filter import LinBoundedParticleFilter
from .manifold_mixins import LinBoundedFilterMixin

class LinPeriodicParticleFilter(LinBoundedParticleFilter, LinBoundedFilterMixin):
    pass
