import numpy as np
from lcd_mcvm_distance_sym import MCvMDistanceSym

class MCvMDistanceSymEven(MCvMDistanceSym):
    def __init__(self, dim, num_half_samples):
        super().__init__(dim, num_half_samples, 1.0 / (2.0 * num_half_samples))

    def get_samples(self):
        return np.vstack((self.half_samples, -self.half_samples))

    def compute_d2(self):
        d2 = super().compute_d2_base()
        return d2

    def compute_d3(self):
        d3 = super().compute_d3_base()
        return d3

    def compute_grad2(self):
        grad2 = super().compute_grad2_base()
        return grad2
