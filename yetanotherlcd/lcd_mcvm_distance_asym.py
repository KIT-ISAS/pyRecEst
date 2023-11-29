import numpy as np
from lcd_mcvm_distance import MCvMDistance
from scipy.integrate import quad

class MCvMDistanceAsym(MCvMDistance):
    def __init__(self, dim: int, num_samples: int):
        super().__init__(dim)
        self.num_samples = num_samples
        self.sample_weight = 1.0 / num_samples
        self.sample_weight_squared = self.sample_weight * self.sample_weight
        self.coeff_d2 = self.sample_weight
        self.coeff_d3 = self.sample_weight_squared
        self.coeff_grad1 = 2.0 * self.sample_weight
        self.coeff_grad2 = 0.5 * self.sample_weight_squared
        self.samples = None
        self.squared_norms = None
        self.grad1_squared_norm = None

    def set_parameters(self, parameters: np.ndarray):
        if not self.check_parameters(parameters):
            raise ValueError("Invalid asymmetric Gaussian mCvM distance parameters.")
        self.samples = parameters
        self.squared_norms = np.sum(parameters**2, axis=0)

    def get_samples(self) -> np.ndarray:
        return self.samples

    def check_parameters(self, parameters: np.ndarray) -> bool:
        if parameters.shape[0] != self.dim or parameters.shape[1] != self.num_samples:
            return False
        else:
            return True

    def compute_d2(self) -> float:
        quad_d2 = lambda b: self.compute_quad_d2(b)
        quad_res, _ = quad(quad_d2, 0, float('inf'))
        return self.coeff_d2 * quad_res

    def compute_d3(self) -> float:
        d3a = 0
        d3b = 0

        for i in range(self.num_samples):
            for j in range(self.num_samples):
                sn_minus = np.sum((self.samples[:, i] - self.samples[:, j]) ** 2)
                csn_minus = self.coeff_squared_norm * sn_minus

                d3a += np.exp(csn_minus)
                d3b += sn_minus * self.exp_int(csn_minus)

        return self.coeff_d3 * (0.5 * self.b_max_squared * d3a + 0.125 * d3b)

    def compute_grad1(self) -> np.ndarray:
        grad1 = zeros((self.dim, self.num_samples))

        for i in range(self.num_samples):
            self.grad1_squared_norm = self.squared_norms[i]
            quad_grad1 = lambda b: self.compute_quad_grad1(b)
            quad_res, _ = quad(quad_grad1, 0, float('inf'))
            grad1[:, i] = self.samples[:, i] * quad_res

        return self.coeff_grad1 * grad1

    def compute_grad2(self) -> np.ndarray:
        grad2 = zeros((self.dim, self.num_samples))

        for e in range(self.num_samples):
            for i in range(self.num_samples):
                minus = self.samples[:, e] - self.samples[:, i]
                csn_minus = self.coeff_squared_norm * np.sum(minus ** 2)
                grad2[:, e] += minus * self.exp_int(csn_minus)

        return self.coeff_grad2 * grad2

    def compute_quad_d2(self, b: float) -> float:
        b_squared = 2.0 * b * b
        b_squared_inv = 1.0 / (1.0 + b_squared)
        coeff = (b_squared * b_squared_inv) ** (self.half_dim) * b
        return coeff * np.sum(np.exp((-0.5 * b_squared_inv) * self.squared_norms))

    def compute_quad_grad1(self, b: float) -> float:
        b_squared = 2.0 * b * b
        b_squared_inv = 1.0 / (1.0 + b_squared)
        coeff = (b_squared * b_squared_inv) ** (self.half_dim) * b * b_squared_inv
        return coeff * np.exp((-0.5 * b_squared_inv) * self.grad1_squared_norm)
