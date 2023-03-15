import numpy as np
from matplotlib.pyplot import plot
from math import sqrt

class WDDistribution:
    def __init__(self, d, w=None):
        self.d = d
        self.w = w

    def pdf(self, xs):
        raise Exception("Pdf is not defined for WDDistribution.")

    def plot(self):
        plot(self.d, self.w, 'r+')

    def plot_interpolated(self, plot_string='-'):
        raise Exception("No interpolation available for WDDistribution.")

    def trigonometric_moment(self, n, whole_range=False, transformation='identity'):
        if not whole_range:
            exponent_part = np.exp(1j * n * self.d)
        else:
            ind_range = np.arange(n + 1).astype(self.d.dtype)
            exponent_part = np.exp(1j * (self.d.reshape(1, -1) * ind_range.reshape(-1, 1)))

        if self.w is None:
            if transformation == 'identity':
                m = exponent_part.mean(axis=-1)
            elif transformation == 'sqrt':
                m = np.sum(exponent_part, axis=1) * sqrt(1 / self.d.shape[0])
            elif transformation == 'log':
                m = np.sum(exponent_part, axis=1).mean(axis=-1)
            else:
                raise Exception("Transformation not supported")
        else:
            m = np.dot(exponent_part, self.w.astype(exponent_part.dtype))
        return m