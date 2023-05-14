from .hypertoroidal_particle_filter import HypertoroidalParticleFilter

class ToroidalParticleFilter(HypertoroidalParticleFilter):
    def __init__(self, n_particles):
        HypertoroidalParticleFilter.__init__(self, n_particles, 2)

