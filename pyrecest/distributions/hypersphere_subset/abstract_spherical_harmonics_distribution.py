from math import pi
from pyrecest.backend import sqrt
from pyrecest.backend import real
from pyrecest.backend import isnan
from pyrecest.backend import imag
from pyrecest.backend import array
from pyrecest.backend import abs
from pyrecest.backend import zeros
import copy
import warnings


from scipy.linalg import norm

from ..abstract_orthogonal_basis_distribution import AbstractOrthogonalBasisDistribution
from .abstract_spherical_distribution import AbstractSphericalDistribution


class AbstractSphericalHarmonicsDistribution(
    AbstractSphericalDistribution, AbstractOrthogonalBasisDistribution
):
    def __init__(self, coeff_mat, transformation="identity"):
        AbstractSphericalDistribution.__init__(self)
        coeff_mat = np.atleast_2d(coeff_mat)
        assert (
            coeff_mat.shape[1] == coeff_mat.shape[0] * 2 - 1
        ), "CoefficientMatrix:Size, Dimensions of coefficient Matrix are incompatible."

        # Ignore irrelevant entries of coeff_mat and set to NaN
        n = coeff_mat.shape[0]
        coeff_mat = coeff_mat + np.block(
            [
                [
                    zeros((n - 1, 1)),
                    np.kron(
                        np.triu(np.full((n - 1, n - 1), float('NaN'))), array([[1, 1]])
                    ),
                ],
                [zeros((1, 2 * n - 1))],
            ]
        )
        AbstractOrthogonalBasisDistribution.__init__(self, coeff_mat, transformation)

    def pdf(self, xs):
        return AbstractOrthogonalBasisDistribution.pdf(self, xs)

    def normalize_in_place(self):
        int_val = self.integrate()
        if int_val < 0:
            warnings.warn(
                "Warning: Normalization:negative - Coefficient for first degree is negative. "
                "This can either be caused by a user error or due to negativity caused by "
                "non-square rooted version"
            )
        elif abs(int_val) < 1e-12:
            raise ValueError(
                "Normalization:almostZero - Coefficient for first degree is too close to zero, "
                "this usually points to a user error"
            )
        elif abs(int_val - 1) > 1e-5:
            warnings.warn(
                "Warning: Normalization:notNormalized - Coefficients apparently do not belong "
                "to normalized density. Normalizing..."
            )
        else:
            return

        if self.transformation == "identity":
            self.coeff_mat = self.coeff_mat / int_val
        elif self.transformation == "sqrt":
            self.coeff_mat = self.coeff_mat / sqrt(int_val)
        else:
            warnings.warn("Warning: Currently cannot normalize")

    def integrate(self):
        if self.transformation == "identity":
            int_val = self.coeff_mat[0, 0] * sqrt(4 * pi)
        elif self.transformation == "sqrt":
            int_val = norm(self.coeff_mat[~isnan(self.coeff_mat)]) ** 2
        else:
            raise ValueError("No analytical formula for normalization available")

        assert abs(imag(int_val) < 1e-8)
        return real(int_val)

    def truncate(self, degree):
        result = copy.deepcopy(self)
        if result.coeff_mat.shape[0] - 1 > degree:
            result.coeff_mat = result.coeff_mat[
                : degree + 1, : 2 * degree + 1
            ]  # noqa: E203
        elif result.coeff_mat.shape[0] - 1 < degree:
            warnings.warn("Less coefficients than desired, filling up with zeros")
            new_coeff_mat = zeros(
                (degree + 1, 2 * degree + 1), dtype=self.coeff_mat.dtype
            )
            new_coeff_mat[
                : result.coeff_mat.shape[0],
                : 2 * result.coeff_mat.shape[0] - 1,  # noqa: E203
            ] = result.coeff_mat
            for i in range(new_coeff_mat.shape[0] - 1):
                new_coeff_mat[i, 2 * i + 1 :] = float('NaN')  # noqa: E203
            result.coeff_mat = new_coeff_mat

        return result