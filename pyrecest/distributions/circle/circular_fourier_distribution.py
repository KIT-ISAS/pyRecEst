from pyrecest.backend import sum
from pyrecest.backend import sqrt
from pyrecest.backend import sin
from pyrecest.backend import real
from pyrecest.backend import linspace
from pyrecest.backend import imag
from pyrecest.backend import hstack
from pyrecest.backend import exp
from pyrecest.backend import cos
from pyrecest.backend import concatenate
from pyrecest.backend import arange
from pyrecest.backend import int64
from pyrecest.backend import int32
import warnings

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from numpy.fft import irfft, rfft

from .abstract_circular_distribution import AbstractCircularDistribution
from .circular_dirac_distribution import CircularDiracDistribution


class CircularFourierDistribution(AbstractCircularDistribution):
    """
    Circular Fourier Distribution. This is based on my implementation for pytorch in pyDirectional
    """

    @beartype
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        transformation: str = "sqrt",
        c: np.ndarray | None = None,
        a: np.ndarray | None = None,
        b: np.ndarray | None = None,
        n: int | None = None,
        multiplied_by_n: bool = True,
    ):
        AbstractCircularDistribution.__init__(self)
        assert (a is None) == (b is None)
        assert (a is None) != (c is None)
        if c is not None:  # Assumed as result from rfft
            assert c.ndim == 1
            self.c = c
            self.a = None
            self.b = None
            if n is None:
                warnings.warn(
                    "It is not clear for complex ones since they may include another coefficient or not (imaginary part of the last coefficient). Assuming it is relevant."
                )
                self.n = 2 * np.size(c) - 1
            else:
                self.n = n
        elif a is not None and b is not None:
            assert a.ndim == 1
            assert b.ndim == 1
            self.a = a
            self.b = b
            self.c = None
            self.n = a.shape[0] + b.shape[0]
            assert self.n == n or n is None
        else:
            raise ValueError("Need to provide either c or a and b.")

        self.multiplied_by_n = multiplied_by_n
        self.transformation = transformation

    @beartype
    def __sub__(
        self, other: "CircularFourierDistribution"
    ) -> "CircularFourierDistribution":
        # If transformed will not yield minus of pdfs!
        assert (
            (self.a is not None and other.a is not None)
            and (self.b is not None and other.b is not None)
        ) or (
            self.c is not None and other.c is not None
        ), "Either both instances should have `a` and `b` defined, or both should have `c` defined."

        assert self.transformation == other.transformation
        assert self.n == other.n
        assert self.multiplied_by_n == other.multiplied_by_n
        if self.a is not None and self.b is not None:
            aNew = self.a - other.a
            bNew = self.b - other.b
            fdNew = CircularFourierDistribution(
                a=aNew,
                b=bNew,
                transformation=self.transformation,
                multiplied_by_n=self.multiplied_by_n,
            )
        else:
            cNew = self.c - other.c
            fdNew = CircularFourierDistribution(
                c=cNew,
                transformation=self.transformation,
                multiplied_by_n=self.multiplied_by_n,
            )
        fdNew.n = (
            self.n
        )  # The number should not change! We store it if we use a complex one now and set it to None if we falsely believe we know the number (it is not clear for complex ones)
        return fdNew

    @beartype
    def pdf(self, xs: np.ndarray) -> np.ndarray:
        assert xs.ndim <= 2, "xs should have at most 2 dimensions."
        xs = xs.reshape(-1, 1)
        a, b = self.get_a_b()

        k_range = arange(1, a.shape[0]).astype(xs.dtype)
        p = a[0] / 2 + sum(
            a[1:].reshape(1, -1) * cos(xs * k_range)
            + b.reshape(1, -1) * sin(xs * k_range),
            axis=1,
        )
        if self.multiplied_by_n:
            p = p / self.n
        if self.transformation == "sqrt":
            p = p**2
        elif self.transformation == "identity":
            pass
        else:
            raise NotImplementedError("Transformation not supported.")
        return p

    def normalize(self) -> "CircularFourierDistribution":
        integral_value = self.integrate()

        if self.a is not None and self.b is not None:
            if self.transformation == "identity":
                scale_factor = 1 / integral_value
            elif self.transformation == "sqrt":
                scale_factor = 1 / sqrt(integral_value)
            else:
                raise NotImplementedError("Transformation not supported.")

            a_new = self.a * scale_factor
            b_new = self.b * scale_factor
            fd_normalized = CircularFourierDistribution(
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
                scale_factor = 1 / sqrt(integral_value)
            else:
                raise NotImplementedError("Transformation not supported.")

            c_new = self.c * scale_factor
            fd_normalized = CircularFourierDistribution(
                c=c_new,
                transformation=self.transformation,
                n=self.n,
                multiplied_by_n=self.multiplied_by_n,
            )

        else:
            raise ValueError("Need either a and b or c.")

        return fd_normalized

    # pylint: disable=too-many-branches
    def integrate(self, integration_boundaries=None) -> float:
        assert (
            integration_boundaries is None
        ), "Currently, only supported for entire domain."
        if self.a is not None and self.b is not None:
            a: np.ndarray = self.a
            b: np.ndarray = self.b
            if self.multiplied_by_n:
                a = a * (1 / self.n)
                b = b * (1 / self.n)

            if self.transformation == "identity":
                a0_non_rooted = a[0]
            elif self.transformation == "sqrt":
                from_a0 = a[0] ** 2 * 0.5
                from_a1_to_end_and_b = sum(a[1:] ** 2) + sum(b**2)
                a0_non_rooted = from_a0 + from_a1_to_end_and_b
            else:
                raise NotImplementedError("Transformation not supported.")
            integral = a0_non_rooted * np.pi
        elif self.c is not None:
            if self.transformation == "identity":
                if self.multiplied_by_n:
                    c0 = real(self.c[0]) * (1 / self.n)
                else:
                    c0 = real(self.c[0])
                integral = 2 * np.pi * c0
            elif self.transformation == "sqrt":
                if self.multiplied_by_n:
                    c = self.c * (1 / self.n)
                else:
                    c = self.c
                from_c0 = (real(c[0])) ** 2
                from_c1_to_end = sum((real(c[1:])) ** 2) + sum(
                    (imag(c[1:])) ** 2
                )

                a0_non_rooted = 2 * from_c0 + 4 * from_c1_to_end
                integral = a0_non_rooted * np.pi
            else:
                raise NotImplementedError("Transformation not supported.")
        else:
            raise ValueError("Need either a and b or c.")
        return integral

    def plot_grid(self):
        grid_values = irfft(self.get_c(), self.n)
        xs = linspace(0, 2 * np.pi, grid_values.shape[0], endpoint=False)
        vals = grid_values.squeeze()

        if self.transformation == "sqrt":
            p = vals**2
        elif self.transformation == "log":
            p = exp(vals)
        elif self.transformation == "identity":
            p = vals
        else:
            raise NotImplementedError("Transformation not supported.")

        plt.plot(xs, p, "r+")
        plt.show()

    @beartype
    def plot(self, resolution=128, **kwargs):
        xs = linspace(0, 2 * np.pi, resolution)

        if self.a is not None:
            xs = xs.astype(self.a.dtype)
        else:
            xs = xs.astype(real(self.c).dtype)

        pdf_vals = self.pdf(xs)

        p = plt.plot(xs, pdf_vals, **kwargs)
        plt.show()

        return p

    def get_a_b(self) -> tuple[np.ndarray, np.ndarray]:
        if self.a is not None:
            a = self.a
            b = self.b
        elif self.c is not None:
            a = 2 * real(self.c)
            b = -2 * imag(self.c[1:])
        assert (
            self.n is None or (np.size(a) + np.size(b)) == self.n
        )  # Other case not implemented yet!
        return a, b

    def get_c(self) -> np.ndarray:
        if self.a is not None:
            c = (self.a[0] + 1j * hstack((0, self.b))) * 0.5
        elif self.c is not None:
            c = self.c
        return c

    def to_real_fd(self):
        if self.a is not None:
            fd = self
        elif self.c is not None:
            a, b = self.get_a_b()
            fd = CircularFourierDistribution(
                transformation=self.transformation,
                a=a,
                b=b,
                n=self.n,
                multiplied_by_n=self.multiplied_by_n,
            )
        return fd

    def get_full_c(self):
        assert self.c is not None
        neg_c = np.conj(
            self.c[-1:0:-1]
        )  # Create array for negative-frequency components
        full_c = concatenate(
            [neg_c, self.c]
        )  # Concatenate arrays to get full spectrum
        return full_c

    @staticmethod
    @beartype
    def from_distribution(
        distribution: AbstractCircularDistribution,
        n: int | int32 | int64,
        transformation: str = "sqrt",
        store_values_multiplied_by_n: bool = True,
    ) -> "CircularFourierDistribution":
        if isinstance(distribution, CircularDiracDistribution):
            fd = CircularFourierDistribution(
                np.conj(distribution.trigonometric_moment(n, whole_range=True))
                / (2 * np.pi),
                transformation,
                multiplied_by_n=False,
            )
            if store_values_multiplied_by_n:
                warnings.warn("Scaling up for WD (this is not recommended).")
                fd.c = fd.c * fd.n
        else:
            xs = linspace(0, 2 * np.pi, n, endpoint=False)
            fvals = distribution.pdf(xs)
            if transformation == "identity":
                pass
            elif transformation == "sqrt":
                fvals = sqrt(fvals)
            else:
                raise NotImplementedError("Transformation not supported.")
            fd = CircularFourierDistribution.from_function_values(
                fvals, transformation, store_values_multiplied_by_n
            )

        return fd

    @staticmethod
    @beartype
    def from_function_values(
        fvals: np.ndarray,
        transformation: str = "sqrt",
        store_values_multiplied_by_n: bool = True,
    ) -> "CircularFourierDistribution":
        c = rfft(fvals)
        if not store_values_multiplied_by_n:
            c = c * (1 / np.size(fvals))

        fd = CircularFourierDistribution(
            c=c,
            transformation=transformation,
            n=np.size(fvals),
            multiplied_by_n=store_values_multiplied_by_n,
        )

        return fd
