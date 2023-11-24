import numpy as np
from scipy.integrate import quad
from lcd_mcvm_distance import MCvMDistance

class MCvMDistanceSym(MCvMDistance):
    def __init__(self, dim, num_half_samples, sample_weight):
        #self.dim = dim
        self.num_half_samples = num_half_samples
        self.sample_weight = sample_weight
        self.sample_weight_squared = sample_weight * sample_weight

        self.coeff_d2 = 2.0 * sample_weight
        self.coeff_d3 = 2.0 * self.sample_weight_squared
        self.coeff_grad1 = 4.0 * sample_weight
        self.coeff_grad2 = self.sample_weight_squared

        self.squared_norms = zeros(num_half_samples)
        self.half_samples = None
        self.grad1_squared_norm = 0
        super().__init__(dim)


    def set_parameters(self, parameters):
        if not self.check_parameters(parameters):
            raise ValueError("Invalid symmetric Gaussian mCvM distance parameters.")
        self.half_samples = parameters
        self.squared_norms = np.sum(np.square(self.half_samples), axis=1)

    def check_parameters(self, parameters):
        return parameters.shape == (self.num_half_samples, self.dim)

    def compute_d2_base(self):
        def quad_d2(b):
            b_squared = 2.0 * b * b
            b_squared_inv = 1.0 / (1.0 + b_squared)
            coeff = (b_squared * b_squared_inv) ** (self.dim / 2) * b
            return coeff * np.sum(np.exp((-0.5 * b_squared_inv) * self.squared_norms))

        res, _ = quad(quad_d2, 0, float('inf'))
        return self.coeff_d2 * res

    def compute_grad1(self):
        grad1 = zeros((self.num_half_samples, self.dim))

        for i in range(self.num_half_samples):
            self.grad1_squared_norm = self.squared_norms[i]

            def quad_grad1(b):
                b_squared = 2.0 * b * b
                b_squared_inv = 1.0 / (1.0 + b_squared)
                coeff = (b_squared * b_squared_inv) ** (self.dim / 2) * b * b_squared_inv
                return coeff * np.exp((-0.5 * b_squared_inv) * self.grad1_squared_norm)

            res, _ = quad(quad_grad1, 0, float('inf'))
            grad1[i, :] = self.half_samples[i, :] * res

        grad1 *= self.coeff_grad1
        return grad1

    def compute_d3_base(self):
        d3a = 0
        d3b = 0

        for i in range(self.num_half_samples):
            for j in range(self.num_half_samples):
                sn_minus = np.sum(np.square(self.half_samples[i, :] - self.half_samples[j, :]))
                sn_plus = np.sum(np.square(self.half_samples[i, :] + self.half_samples[j, :]))

                csn_minus = self.sample_weight_squared * sn_minus
                csn_plus = self.sample_weight_squared * sn_plus

                d3a += np.exp(csn_minus) + np.exp(csn_plus)
                d3b += sn_minus * np.exp(csn_minus) + sn_plus * np.exp(csn_plus)

        return self.coeff_d3 * (0.5 * self.sample_weight_squared * d3a + 0.125 * d3b)

    def compute_grad2_base(self):
        grad2 = zeros((self.dim, self.num_half_samples))

        for e in range(self.num_half_samples):
            for i in range(self.num_half_samples):
                minus = self.half_samples[e, :] - self.half_samples[i, :]
                plus = self.half_samples[e, :] + self.half_samples[i, :]

                csn_minus = self.sample_weight_squared * np.sum(np.square(minus))
                csn_plus = self.sample_weight_squared * np.sum(np.square(plus))

                grad2[:, e] += minus * np.exp(csn_minus) + plus * np.exp(csn_plus)

        grad2 *= self.coeff_grad2
        return grad2.T
