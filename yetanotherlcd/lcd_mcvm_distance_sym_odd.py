import numpy as np
from scipy.integrate import quad
from lcd_mcvm_distance_sym import MCvMDistanceSym
from lcd_mcvm_distance import MCvMDistance

class MCvMDistanceSymOdd(MCvMDistanceSym):
    def __init__(self, dim, num_half_samples):
        super().__init__(dim, num_half_samples, 1.0 / (2.0 * num_half_samples + 1.0))
        self.tmp_squared_norms = zeros(num_half_samples)
        self.exp_squared_norms = zeros(num_half_samples)
        self.exp_int_squared_norms = zeros(num_half_samples)

    def set_b_max(self, b_max):
        super().set_b_max(b_max)

        quad_const_d2, _ = quad(self.compute_quad_const_d2, 0, b_max)
        self.const_d2 = self.sample_weight * quad_const_d2
        self.const_d3 = self.sample_weight_squared * 0.5 * b_max ** 2

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.tmp_squared_norms = self.coeff_squared_norm * self.squared_norms
        self.exp_squared_norms = np.exp(self.tmp_squared_norms)
        self.exp_int_squared_norms = MCvMDistance.exp_int(self.tmp_squared_norms)

    def get_samples(self):
        """ We get the samples from half_samples and change the convetion dom (d, n) to (n, d)"""
        samples = zeros((2 * self.num_half_samples + 1, self.dim))
        j = 0
        
        for i in range(self.num_half_samples):
            samples[j, :] = self.half_samples[i, :]
            j += 1
            samples[j, :] = -self.half_samples[i, :]
            j += 1

        # Put last sample to center
        samples[j, :] = 0

        return samples


    def compute_d2(self):
        d2 = super().compute_d2_base()
        return d2 + self.const_d2

    def compute_d3(self):
        d3 = super().compute_d3_base()
        d3 += 4.0 * self.sample_weight_squared * (0.5 * self.b_max_squared * self.exp_squared_norms +
                                                  0.125 * self.squared_norms * self.exp_int_squared_norms).sum()
        return d3 + self.const_d3

    def compute_grad2(self):
        grad2 = super().compute_grad2_base()
        return grad2 + self.sample_weight_squared * (np.expand_dims(self.exp_int_squared_norms, -1) * self.half_samples)

    def compute_quad_const_d2(self, b):
        b_squared = 2.0 * b * b
        return (b_squared / (1.0 + b_squared)) ** self.half_dim * b
