import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from numpy.fft import irfft, rfft

from .wd_distribution import WDDistribution

"""
Fourier Distribution. The real and complex ones and the option to have it multiplied_by_n or not serve to use the minimum number of operations and thus optimize the graph
"""


class FourierDistribution:
    def __init__(
        self,
        transformation="sqrt",
        c=None,
        a=None,
        b=None,
        n=None,
        multiplied_by_n=True,
    ):
        assert (a is None) == (b is None)
        assert (a is None) != (c is None)
        if c is not None:
            self.c = c  # Assumed as result from rfft
            self.a = None
            self.b = None
            self.n = None  # It is not clear for complex ones since they may include another coefficient or not (imaginary part of the last coefficient)
        elif a is not None and b is not None:
            self.a = a
            self.b = b
            self.c = None
            self.n = a.shape[0] + b.shape[0]
        else:
            raise Exception("Need to provide either c or a and b.")

        self.multiplied_by_n = multiplied_by_n
        self.transformation = transformation
        if n is not None:
            if a is not None:
                assert self.n == n
            else:
                self.n = n

    def __sub__(self, other):
        # If transformed will not yield minus of pdfs!
        assert (
            self.a is not None
            and other.a is not None
            or self.c is not None
            and other.c is not None
        )
        assert self.transformation == other.transformation
        assert self.n == other.n
        assert self.multiplied_by_n == other.multiplied_by_n
        if self.a is not None:
            aNew = self.a - other.a
            bNew = self.b - other.b
            fdNew = FourierDistribution(
                a=aNew,
                b=bNew,
                transformation=self.transformation,
                multiplied_by_n=self.multiplied_by_n,
            )
        else:
            cNew = self.c - other.c
            fdNew = FourierDistribution(
                c=cNew,
                transformation=self.transformation,
                multiplied_by_n=self.multiplied_by_n,
            )
        fdNew.n = (
            self.n
        )  # The number should not change! We store it if we use a complex one now and set it to None if we falsely believe we know the number (it is not clear for complex ones)
        return fdNew

    def pdf(self, xs):
        assert xs.ndim <= 2
        xs = xs.reshape(-1, 1)
        a, b = self.get_a_b()

        range = np.arange(1, a.shape[0]).astype(xs.dtype)
        p = a[0] / 2 + np.sum(
            a[1:].reshape(1, -1) * np.cos(xs * range)
            + b.reshape(1, -1) * np.sin(xs * range),
            axis=1,
        )
        if self.multiplied_by_n:
            p = p / self.n
        if self.transformation == "sqrt":
            p = p**2
        elif self.transformation == "identity":
            pass
        else:
            raise Exception("Transformation not supported.")
        return p

    def normalize(self):
        integral_value = self.integral()

        if self.a is not None and self.b is not None:
            if self.transformation == "identity":
                scale_factor = 1 / integral_value
            elif self.transformation == "sqrt":
                scale_factor = 1 / np.sqrt(integral_value)
            else:
                raise Exception("Transformation not supported.")

            a_new = self.a * scale_factor
            b_new = self.b * scale_factor
            fd_normalized = FourierDistribution(
                a=a_new,
                b=b_new,
                transformation=self.transformation,
                n=self.n,
                multiplied_by_n=self.multiplied_by_n,
            )

        elif self.c is not None:
            if self.transformation == "identity":
                scale_factor = 1 / integral_value
            elif self.transformation == "sqrt":
                scale_factor = 1 / np.sqrt(integral_value)
            else:
                raise Exception("Transformation not supported.")

            c_new = self.c * scale_factor
            fd_normalized = FourierDistribution(
                c=c_new,
                transformation=self.transformation,
                n=self.n,
                multiplied_by_n=self.multiplied_by_n,
            )

        else:
            raise Exception("Need either a and b or c.")

        return fd_normalized

    def integral(self):
        pi = np.pi
        if self.a is not None and self.b is not None:
            if self.multiplied_by_n:
                a = self.a * (1 / self.n)
                b = self.b * (1 / self.n)
            else:
                a = self.a
                b = self.b
            if self.transformation == "identity":
                a0_non_rooted = a[0]
            elif self.transformation == "sqrt":
                from_a0 = a[0] ** 2 * 0.5
                from_a1_to_end_and_b = np.sum(a[1:] ** 2) + np.sum(b**2)
                a0_non_rooted = from_a0 + from_a1_to_end_and_b
            else:
                raise Exception("Transformation not supported.")
            integral = a0_non_rooted * pi
        elif self.c is not None:
            if self.transformation == "identity":
                if self.multiplied_by_n:
                    c0 = np.real(self.c[0]) * (1 / self.n)
                else:
                    c0 = np.real(self.c[0])
                integral = 2 * pi * c0
            elif self.transformation == "sqrt":
                if self.multiplied_by_n:
                    c = self.c * (1 / self.n)
                else:
                    c = self.c
                from_c0 = (np.real(c[0])) ** 2
                from_c1_to_end = np.sum((np.real(c[1:])) ** 2) + np.sum(
                    (np.imag(c[1:])) ** 2
                )

                a0_non_rooted = 2 * from_c0 + 4 * from_c1_to_end
                integral = a0_non_rooted * pi
            else:
                raise Exception("Transformation not supported.")
        else:
            raise Exception("Need either a and b or c.")
        return integral

    def integral_numerical(self):
        pi = np.pi
        if self.a is not None:
            a = self.a
        else:
            a = 2 * np.real(self.c)

        integral, _ = integrate.quad(
            lambda x: self.pdf(np.array(x).astype(a.dtype).reshape(1, -1)), 0, 2 * pi
        )
        return integral

    def plot_grid(self):
        grid_values = irfft(self.get_c(), self.n)
        xs = np.linspace(0, 2 * np.pi, grid_values.shape[0], endpoint=False)
        vals = grid_values.squeeze()

        if self.transformation == "sqrt":
            p = vals**2
        elif self.transformation == "log":
            p = np.exp(vals)
        elif self.transformation == "identity":
            p = vals
        else:
            raise Exception("Transformation not supported.")

        plt.plot(xs, p, "r+")
        plt.show()

    def plot(self, plot_string="-", **kwargs):
        xs = np.linspace(0, 2 * np.pi, 100).reshape([100, 1])

        if self.a is not None:
            xs = xs.astype(self.a.dtype)
        else:
            xs = xs.astype(np.real(self.c).dtype)

        pdf_vals = self.pdf(xs)

        plt.plot(xs, pdf_vals, plot_string, **kwargs)
        plt.show()

        return np.max(pdf_vals)

    def get_a_b(self):
        if self.a is not None:
            a = self.a
            b = self.b
        elif self.c is not None:
            a = 2 * np.real(self.c)
            b = -2 * np.imag(self.c[1:])
        assert (
            self.n is None or (a.size + b.size) == self.n
        )  # Other case not implemented yet!
        return a, b

    def get_c(self):
        if self.a is not None:
            c = (self.a[0] + 1j * np.hstack((0, self.b))) * 0.5
        elif self.c is not None:
            c = self.c
        return c

    def to_real_fd(self):
        if self.a is not None:
            fd = self
        elif self.c is not None:
            a, b = self.get_a_b()
            fd = FourierDistribution(
                transformation=self.transformation,
                a=a,
                b=b,
                n=self.n,
                multiplied_by_n=self.multiplied_by_n,
            )
        return fd

    @staticmethod
    def from_distribution(dist, n, transformation, store_values_multiplied_by_n=True):
        if isinstance(dist, WDDistribution):
            fd = FourierDistribution(
                np.conj(dist.trigonometric_moment(n, whole_range=True)) / (2 * np.pi),
                transformation,
                multiplied_by_n=False,
            )
            if store_values_multiplied_by_n:
                warnings.warn("Scaling up for WD (this is not recommended).")
                fd.c = fd.c * fd.n
        else:
            xs = np.linspace(0, 2 * np.pi, n + 1).astype(dist.get_params()[0].dtype)
            fvals = dist.pdf(xs[:-1])
            if transformation == "identity":
                pass
            elif transformation == "sqrt":
                fvals = np.sqrt(fvals)
            else:
                raise Exception("Transformation not supported.")
            c = rfft(fvals)
            if not store_values_multiplied_by_n:
                c = c * (1 / n)

            fd = FourierDistribution(
                c=c,
                transformation=transformation,
                n=n,
                multiplied_by_n=store_values_multiplied_by_n,
            )
        return fd
