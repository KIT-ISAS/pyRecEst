from abc import abstractmethod
from math import pi

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import mod, ndim, sin
from scipy.special import ive  # pylint: disable=no-name-in-module
from scipy.stats import vonmises

from .abstract_circular_distribution import AbstractCircularDistribution
from .wrapped_cauchy_distribution import WrappedCauchyDistribution
from .wrapped_normal_distribution import WrappedNormalDistribution


class GeneralizedKSineSkewedVonMisesDistribution(AbstractCircularDistribution):
    # See Bekker, A., Nakhaei Rad, N., Arashi, M., Ley, C. (2020). Generalized Skew-Symmetric Circular and
    # Toroidal Distributions, Florence Nightingale Directional Statistics volume, Springer.
    def __init__(self, mu, kappa, lambda_, k, m):
        AbstractCircularDistribution.__init__(self)
        self.mu = mod(mu, 2 * pi)
        self.kappa = kappa
        self.lambda_ = lambda_
        self.k = k
        self.m = m

        self.validate_parameters()

    def validate_parameters(self):
        assert -1.0 <= self.lambda_ and self.lambda_ <= 1.0
        assert isinstance(self.m, int) and self.m >= 1

    def pdf(self, xs):
        # Evaluate the von Mises distribution and multiply by (1 + lambda_ * sin(xa - mu))
        assert self.k == 1, "Currently, only k=1 is supported"
        vm_pdf = vonmises.pdf(xs, self.kappa, loc=self.mu)
        skew_factor = (1 + self.lambda_ * sin(self.k * (xs - self.mu))) ** self.m
        if self.m == 1:
            norm_const = 1
        elif self.m == 2:
            norm_const = 1 / (
                1 + self.lambda_**2 / 2 * (1 - bessel_ratio(2, self.kappa))
            )
        elif self.m == 3:
            norm_const = 1 / (
                1 + 3 * self.lambda_**2 / 2 * (1 - bessel_ratio(2, self.kappa))
            )
        elif self.m == 4:
            norm_const = 1 / (
                1
                + self.lambda_**4
                / 8
                * (3 - 2 * bessel_ratio(2, self.kappa) + bessel_ratio(4, self.kappa))
                + 3 * self.lambda_**2 * (1 - bessel_ratio(2, self.kappa))
            )
        else:
            raise NotImplementedError("m > 4 not implemented")

        return norm_const * vm_pdf * skew_factor

    def shift(self, shift_by):
        if ndim(shift_by) != 0:
            raise ValueError("angle must be a scalar")
        new_dist = GeneralizedKSineSkewedVonMisesDistribution(
            self.mu + shift_by, self.kappa, self.lambda_, self.k, self.m
        )
        return new_dist


class SineSkewedVonMisesDistribution(GeneralizedKSineSkewedVonMisesDistribution):
    def __init__(self, mu, kappa, lambda_):
        super().__init__(mu, kappa, lambda_, k=1, m=1)


def bessel_ratio(p, z):
    """
    Computes the ratio I_p(z) / I_p(0) in a numerically stable manner using
    exponentially scaled modified Bessel functions.

    Parameters:
    - p: Order of the Bessel function.
    - z: Argument for the Bessel function.

    Returns:
    - The ratio of I_p(z) to I_p(0), calculated in a numerically stable way.
    """
    # Compute the scaled Bessel function values for both the numerator and denominator.
    scaled_numerator = ive(p, z)
    scaled_denominator = ive(p, 0)

    # Since ive(p, z) = iv(p, z) * exp(-|z|), and ive(p, 0) = iv(p, 0),
    # when we take the ratio, the exp(-|z|) terms cancel out for the ratio calculation.
    # Therefore, the ratio of the scaled values directly gives us the ratio of the original Bessel functions.
    return scaled_numerator / scaled_denominator


class AbstractSineSkewedDistribution(AbstractCircularDistribution):
    """
    Abstract superclass for sine-skewed distributions.
    """

    def __init__(self, mu, lambda_):
        """
        Initialize the sine-skewed distribution with a central location parameter mu
        and a skewness parameter lambda_.
        """
        AbstractCircularDistribution.__init__(self)
        self.mu = mu
        self.lambda_ = lambda_

    @abstractmethod
    def base_pdf(self, xs):
        """
        Compute the base probability density function (PDF) for the wrapped distribution
        without skewness. This method must be implemented by subclasses.
        """

    def pdf(self, xs):
        """
        Compute the skewed probability density function (PDF) for the distribution.
        """
        # Calculate the base pdf from the wrapped distribution
        base_pdf = self.base_pdf(xs)

        # Apply the skewing factor
        skewed_pdf = base_pdf * (1 + self.lambda_ * sin(xs - self.mu))

        return skewed_pdf


class SineSkewedWrappedNormalDistribution(AbstractSineSkewedDistribution):
    def __init__(self, mu, sigma, lambda_):
        super().__init__(mu, lambda_)
        self.wrapped_normal = WrappedNormalDistribution(mu, sigma)

    @property
    def sigma(self):
        return self.wrapped_normal.sigma

    def base_pdf(self, xs):
        return self.wrapped_normal.pdf(xs)


class SineSkewedWrappedCauchyDistribution(AbstractSineSkewedDistribution):
    def __init__(self, mu, gamma, lambda_):
        super().__init__(mu, lambda_)
        self.wrapped_cauchy = WrappedCauchyDistribution(mu, gamma)

    @property
    def gamma(self):
        return self.wrapped_cauchy.gamma

    def base_pdf(self, xs):
        return self.wrapped_cauchy.pdf(xs)
