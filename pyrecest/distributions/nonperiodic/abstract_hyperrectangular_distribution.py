import numpy as np
from ..abstract_bounded_nonperiodic_distribution import AbstractBoundedNonPeriodicDistribution
import scipy

class AbstractHyperrectangularDistribution(AbstractBoundedNonPeriodicDistribution):
    def __init__(self, bounds):
        AbstractBoundedNonPeriodicDistribution.__init__(self, np.size(bounds[0]))
        self.bounds = bounds
        
    def get_manifold_size(self):
        s = np.prod(np.diff(self.bounds, axis=1))
        return s
    
    @property
    def input_dim(self):
        return self.dim

    def integrate(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.bounds

        left = np.atleast_1d(integration_boundaries[0])
        right = np.atleast_1d(integration_boundaries[1])

        def pdf_fun(*args):
            return self.pdf(np.array(args))

        integration_boundaries = [(left[i], right[i]) for i in range(self.dim)]
        return scipy.integrate.nquad(pdf_fun, integration_boundaries)[0]