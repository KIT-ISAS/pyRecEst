from numpy.random import standard_normal
import numpy as np
from lcd_mcvm_distance import MCvMDistance
from lcd_mcvm_distance_asym import MCvMDistanceAsym
from lcd_mcvm_distance_sym_even import MCvMDistanceSymEven
from lcd_mcvm_distance_sym_odd import MCvMDistanceSymOdd
from lcd_mcvm_distance_optimizer import MCvMDistanceOptimizer
from lcd_misc import mean_correction, covariance_correction

class Computation:
    def __init__(self):
        self.set_symmetric(True)
        self.set_b_max(-1)

    def set_symmetric(self, use_symmetric):
        self.use_symmetric = use_symmetric

    def set_b_max(self, b_max):
        self.force_b_max = b_max

    def __call__(self, dimension, num_samples, initial_parameters, dist_corrected_samples=None):
        if dimension <= 0:
            raise ValueError("The dimension must be greater than zero.")

        if self.use_symmetric:
            if num_samples < 2 * dimension:
                raise ValueError("The number of samples μst be at least twice the dimension.")

            num_half_samples = num_samples // 2
            is_even = (num_samples % 2) == 0

            if is_even:
                distance = MCvMDistanceSymEven(dimension, num_half_samples)
            else:
                distance = MCvMDistanceSymOdd(dimension, num_half_samples)

            if initial_parameters.size == 0:
                initial_parameters = standard_normal((num_half_samples, dimension))
        else:
            if num_samples <= dimension:
                raise ValueError("The number of samples must be greater than the dimension.")

            distance = MCvMDistanceAsym(dimension, num_samples)

            if initial_parameters.size == 0:
                initial_parameters = standard_normal((num_samples, dimension))

        if self.force_b_max < 0:
            if dimension == 1:
                b_max = 5
            elif dimension <= 10:
                b_max = 10
            elif dimension <= 100:
                b_max = 50
            elif dimension <= 1000:
                b_max = 100
            else:
                b_max = 200
        else:
            b_max = self.force_b_max

        
        distance.set_b_max(b_max)
        
        optimizer = MCvMDistanceOptimizer()
        
        result = optimizer(distance, initial_parameters)
        
        if not result.success:
            raise RuntimeError("Optimizer failed.")
            
        if self.use_symmetric:
            samples = covariance_correction(distance.get_samples())
        else:
            samples = mean_correction(distance.get_samples())
            samples = covariance_correction(samples)
        
        if not np.isfinite(samples).all():
            raise RuntimeError("Computed invalid sample positions.")
        
        if dist_corrected_samples is not None:
            dist = MCvMDistance(dimension=distance.dimension, num_samples=distance.num_samples, symmetric=False)
            dist.set_b_max(b_max)
            dist.set_parameters(samples)
            dist.compute(dist_corrected_samples)
        
        return samples, result
