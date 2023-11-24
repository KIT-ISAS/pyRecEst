import numpy as np
from lcd_mcvm_distance import MCvMDistance
from scipy.optimize import minimize


class MCvMDistanceOptimizer:
    def __init__(self):
        self.history_size = 10
        self.max_num_iterations = 10000
        self.gradient_tol = 1e-8
        self.relative_tol = 1e-6
        self.distance = None
        self.rows = None
        self.cols = None

    def __call__(self, distance: MCvMDistance, initial_parameters: np.ndarray):
        if distance is None:
            raise ValueError("Invalid mCvM distance pointer.")
        
        self.distance = distance
        self.rows = initial_parameters.shape[0]
        self.cols = initial_parameters.shape[1]

        def objective_function(para: np.ndarray):
            parameters = para.reshape(self.rows, self.cols)
            
            self.distance.set_parameters(parameters)
            func_value = self.distance.compute()
            gradient = self.distance.compute_gradient()
            
            return func_value, gradient.ravel()

        result = minimize(fun=objective_function,
                          x0=initial_parameters.ravel(),
                          method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': self.max_num_iterations,
                                   'ftol': self.relative_tol,
                                   'gtol': self.gradient_tol})
        
        return result

    def set_history_size(self, history_size: int):
        self.history_size = history_size

    def get_history_size(self) -> int:
        return self.history_size

    def set_max_num_iterations(self, num_iterations: int):
        self.max_num_iterations = num_iterations

    def get_max_num_iterations(self) -> int:
        return self.max_num_iterations

    def set_gradient_tol(self, grad_tol: float):
        self.gradient_tol = grad_tol

    def get_gradient_tol(self) -> float:
        return self.gradient_tol

    def set_relative_tol(self, rel_tol: float):
        self.relative_tol = rel_tol

    def get_relative_tol(self) -> float:
        return self.relative_tol
