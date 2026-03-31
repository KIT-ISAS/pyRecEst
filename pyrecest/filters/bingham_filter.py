# pylint: disable=no-name-in-module,no-member
import copy

from pyrecest.backend import array, diag
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)

from .abstract_filter import AbstractFilter


class BinghamFilter(AbstractFilter):
    """Recursive filter based on the Bingham distribution.

    Supports antipodally symmetric complex numbers (2D) and quaternions (4D).

    References:
    - Gerhard Kurz, Igor Gilitschenski, Simon Julier, Uwe D. Hanebeck,
      Recursive Bingham Filter for Directional Estimation Involving 180
      Degree Symmetry, Journal of Advances in Information Fusion,
      9(2):90-105, December 2014.
    - Igor Gilitschenski, Gerhard Kurz, Simon J. Julier, Uwe D. Hanebeck,
      Unscented Orientation Estimation Based on the Bingham Distribution,
      IEEE Transactions on Automatic Control, January 2016.
    """

    def __init__(self):
        # Default 4-D identity initial state (uniform on S^3, suitable for quaternion orientation)
        initial_state = BinghamDistribution(
            array([-1.0, -1.0, -1.0, 0.0]),
            array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),
        )
        AbstractFilter.__init__(self, initial_state)

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        assert isinstance(
            new_state, BinghamDistribution
        ), "filter_state must be a BinghamDistribution"
        assert new_state.dim in (
            1,
            3,
        ), "Only 2D and 4D Bingham distributions are supported"
        self._filter_state = copy.deepcopy(new_state)

    def predict_identity(self, bw):
        """Predict assuming identity system model with Bingham noise.

        Computes x(k+1) = x(k) (*) w(k) where (*) is complex or quaternion
        multiplication and w(k) ~ bw.

        Parameters:
            bw (BinghamDistribution): noise distribution
        """
        assert isinstance(bw, BinghamDistribution)
        self.filter_state = self.filter_state.compose(bw)

    def predict_nonlinear(self, a, bw):
        """Predict assuming nonlinear system model with Bingham noise.

        Computes x(k+1) = a(x(k)) (*) w(k) using a sigma-point approximation.

        Parameters:
            a (callable): nonlinear system function mapping R^n -> R^n
            bw (BinghamDistribution): noise distribution
        """
        assert isinstance(bw, BinghamDistribution)

        samples, weights = self.filter_state.sample_deterministic(0.5)

        # Propagate each sample through the system function
        for i in range(len(weights)):
            samples[:, i] = a(samples[:, i])

        # Compute scatter matrix of propagated samples
        S = samples @ diag(weights) @ samples.T
        S = (S + S.T) / 2

        predicted = BinghamDistribution.fit_to_moment(S)
        self.filter_state = predicted.compose(bw)

    def update_identity(self, bv, z):
        """Update assuming identity measurement model with Bingham noise.

        Applies the measurement z using likelihood based on Bingham noise bv.

        Parameters:
            bv (BinghamDistribution): measurement noise distribution
            z (numpy.ndarray): measurement as a unit vector of shape (dim+1,)
        """
        assert isinstance(bv, BinghamDistribution)
        assert bv.dim == self.filter_state.dim
        assert z.shape == (self.filter_state.input_dim,)

        bv = copy.deepcopy(bv)
        n = bv.input_dim
        for i in range(n):
            m_conj = self._conjugate(bv.M[:, i])
            bv.M[:, i] = self._compose(z, m_conj)

        self.filter_state = self.filter_state.multiply(bv)

    def get_point_estimate(self):
        """Return the mode of the current distribution as a point estimate."""
        return self.filter_state.mode()

    @staticmethod
    def _conjugate(q):
        """Return the conjugate of a unit complex number or quaternion.

        For q = [w, x, y, z], conjugate = [w, -x, -y, -z].
        For q = [a, b], conjugate = [a, -b].
        """
        result = q.copy()
        result[1:] = -result[1:]
        return result

    @staticmethod
    def _compose(q1, q2):
        """Compose two unit complex numbers or quaternions via multiplication.

        Parameters:
            q1, q2: unit vectors of length 2 or 4

        Returns:
            product q1 * q2
        """
        if q1.shape[0] == 2:
            # Complex multiplication
            return array(
                [
                    q1[0] * q2[0] - q1[1] * q2[1],
                    q1[0] * q2[1] + q1[1] * q2[0],
                ]
            )
        # Hamilton quaternion product
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
