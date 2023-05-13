from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)
import numpy as np
from scipy.spatial.transform import Rotation as R

class AbstractSphereSubsetDistribution(AbstractHypersphereSubsetDistribution):
    def __init__(self):
        AbstractHypersphereSubsetDistribution.__init__(self, dim=2)

    @staticmethod
    def sph2cart(azimuth, elevation):
        assert azimuth.ndim == 1 and elevation.ndim == 1
        x, y, z = R.from_euler('ZYX', np.column_stack((azimuth, elevation, np.zeros(np.size(azimuth))))).apply(np.array([1, 0, 0]).reshape(1, 3)).T
        return x, y, -z
    
    @staticmethod
    def cart2sph(x, y, z):
        return AbstractSphereSubsetDistribution.cart2sph_colatitude(x, y, z)
        
    @staticmethod
    def cart2sph_colatitude(x, y, z):
        hxy = np.hypot(x, y)
        colatitude = np.arctan2(hxy, z)
        azimuth = np.arctan2(y, x)
        return azimuth, colatitude
    
    @staticmethod
    def cart2sph_elevation(x, y, z):
        hxy = np.hypot(x, y)
        elevation = np.arctan2(z, hxy)
        azimuth = np.arctan2(y, x)
        return azimuth, elevation
