from abc import ABC, abstractmethod
import numpy as np
from scipy.special import expi
from scipy.integrate import quad
from typing import Callable


class MCvMDistance(ABC):
    def __init__(self, dim: int):
        self.dim = dim
        self.half_dim = dim * 0.5
        self.quadrature_tol = 1e-10
        self.set_b_max(50)

    def set_b_max(self, b: float):
        if b <= 0.0:
            raise ValueError("bMax μst be greater than zero.")

        if not np.isfinite(b):
            raise ValueError("bMax is NaN or Inf.")

        self.b_max = b
        self.b_max_squared = b ** 2
        self.coeff_squared_norm = -1.0 / (4.0 * self.b_max_squared)
        self.compute_d1()

    def get_b_max(self) -> float:
        return self.b_max

    def set_quadrature_tol(self, quad_tol: float):
        if quad_tol <= 0.0:
            raise ValueError("Quadrature tolerance must be greater than zero.")
        self.quadrature_tol = quad_tol

    def compute(self) -> float:
        try:
            d2 = self.compute_d2()
            d3 = self.compute_d3()
            distance = self.d1 - 2.0 * d2 + d3
            return distance
        except Exception as ex:
            raise RuntimeError("Computing mCvM distance failed.") from ex

    def compute_gradient(self) -> np.ndarray:
        try:
            grad1 = self.compute_grad1()
            grad2 = self.compute_grad2()
            gradient = grad1 + grad2
            return gradient
        except Exception as ex:
            raise RuntimeError("Computing mCvM distance gradient failed.") from ex

    @abstractmethod
    def set_parameters(self, parameters: np.ndarray):
        pass

    @abstractmethod
    def get_samples(self) -> np.ndarray:
        pass

    @abstractmethod
    def check_parameters(self, parameters: np.ndarray) -> bool:
        pass

    @abstractmethod
    def compute_d2(self) -> float:
        pass

    @abstractmethod
    def compute_d3(self) -> float:
        pass

    @abstractmethod
    def compute_grad1(self) -> np.ndarray:
        pass

    @abstractmethod
    def compute_grad2(self) -> np.ndarray:
        pass

    def integrate(self, function: Callable[[float], float]) -> float:
        result, error = quad(function, 0, self.b_max, epsrel=self.quadrature_tol)
        return result

    @staticmethod
    def exp_int(x: np.ndarray) -> np.ndarray:
        result = zeros_like(x)
        non_zero_indices = x != 0
        result[non_zero_indices] = expi(x[non_zero_indices])
        return result

    def compute_d1(self):
        def quad_d1(b: float) -> float:
            b_squared = b ** 2
            return (b_squared / (1.0 + b_squared)) ** self.half_dim * b

        try:
            self.d1 = self.integrate(quad_d1)
        except Exception as ex:
            raise RuntimeError("Computing D1 failed.") from ex
