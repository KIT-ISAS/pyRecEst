"""Factorized GIW random-matrix extended object tracker."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    asarray,
    concatenate,
    cos,
    eye,
    linalg,
    linspace,
    maximum,
    mean,
    pi,
    sin,
)
from pyrecest.utils.plotting import plot_ellipsoid

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class FactorizedGIWRandomMatrixTracker(AbstractExtendedObjectTracker):
    """Random-matrix tracker using the factorized GIW parameterization.

    This implements the factorized Gaussian inverse-Wishart recursion from
    Granstrom and Bramstang, "Bayesian Smoothing for the Extended Object
    Random Matrix Model", IEEE TSP 2019, Tables V and VI, for constant matrix
    extent dynamics. The public ``extent`` property returns the inverse-Wishart
    mean ``V / (v - 2d - 2)`` while the internal state keeps the natural
    parameters ``v`` and ``V`` explicitly.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        kinematic_state,
        covariance,
        extent_dof,
        extent_scale,
        kinematic_state_to_pos_matrix=None,
        extent_transition_dof=100.0,
        extent_transition_matrix=None,
        measurement_spread_factor=1.0,
        minimum_extent_eigenvalue=1e-12,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extent=False,
        log_posterior_extent=False,
    ):
        super().__init__(
            log_prior_estimates,
            log_posterior_estimates,
            log_prior_extent,
            log_posterior_extent,
        )
        self.kinematic_state = asarray(kinematic_state).reshape(-1)
        self.covariance = asarray(covariance)
        self.extent_dof = float(extent_dof)
        self.extent_scale = self._project_symmetric_positive(
            asarray(extent_scale), minimum_extent_eigenvalue
        )
        self.kinematic_state_to_pos_matrix = kinematic_state_to_pos_matrix
        self.extent_transition_dof = float(extent_transition_dof)
        self.extent_transition_matrix = (
            eye(self.extent_dimension)
            if extent_transition_matrix is None
            else asarray(extent_transition_matrix)
        )
        self.measurement_spread_factor = float(measurement_spread_factor)
        self.minimum_extent_eigenvalue = float(minimum_extent_eigenvalue)
        self._validate_parameters()

    @property
    def extent_dimension(self):
        return int(self.extent_scale.shape[0])

    @property
    def extent_mean_denominator(self) -> float:
        return max(
            self.extent_dof - 2.0 * float(self.extent_dimension) - 2.0,
            self.minimum_extent_eigenvalue,
        )

    @property
    def extent(self):
        return self.extent_scale / self.extent_mean_denominator

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @classmethod
    def _project_symmetric_positive(cls, matrix, minimum_eigenvalue=1e-12):
        matrix = cls._symmetrize(asarray(matrix))
        eigenvalues, eigenvectors = linalg.eigh(matrix)
        if float(eigenvalues[0]) >= minimum_eigenvalue:
            return matrix
        eigenvalues = maximum(eigenvalues, minimum_eigenvalue)
        return cls._symmetrize((eigenvectors * eigenvalues) @ eigenvectors.T)

    def _validate_parameters(self):
        if self.extent_scale.shape[0] != self.extent_scale.shape[1]:
            raise ValueError("extent_scale must be square.")
        if self.extent_dof <= 2.0 * self.extent_dimension + 2.0:
            raise ValueError("extent_dof must be larger than 2 * extent_dimension + 2.")
        if self.extent_transition_dof <= self.extent_dimension + 1.0:
            raise ValueError(
                "extent_transition_dof must be larger than extent_dimension + 1."
            )
        if self.measurement_spread_factor < 0.0:
            raise ValueError("measurement_spread_factor must be non-negative.")
        if self.minimum_extent_eigenvalue <= 0.0:
            raise ValueError("minimum_extent_eigenvalue must be positive.")

    def get_point_estimate(self):
        return concatenate([self.kinematic_state, self.extent.flatten()])

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        if flatten_matrix:
            return self.extent.flatten()
        return self.extent

    # pylint: disable=too-many-arguments,too-many-locals
    def predict(
        self,
        dt,
        Cw,
        tau=None,
        system_matrix=None,
        extent_transition_matrix=None,
        extent_transition_dof=None,
    ):
        """Predict kinematics and GIW extent parameters.

        ``dt`` and ``tau`` are accepted for API compatibility with
        :class:`RandomMatrixTracker`; the factorized GIW recursion uses the
        finite extent transition dof instead.
        """

        del dt, tau
        state_dim = self.kinematic_state.shape[0]
        F = eye(state_dim) if system_matrix is None else asarray(system_matrix)
        Cw = asarray(Cw)
        if Cw.shape in ((), (1,)):
            Cw = Cw * eye(state_dim)
        else:
            Cw = asarray(Cw)

        self.kinematic_state = F @ self.kinematic_state
        self.covariance = self._symmetrize(F @ self.covariance @ F.T + Cw)

        extent_dim = self.extent_dimension
        transition_dof = (
            self.extent_transition_dof
            if extent_transition_dof is None
            else float(extent_transition_dof)
        )
        if transition_dof <= extent_dim + 1.0:
            raise ValueError(
                "extent_transition_dof must be larger than extent_dimension + 1."
            )
        A = (
            self.extent_transition_matrix
            if extent_transition_matrix is None
            else asarray(extent_transition_matrix)
        )

        prior_dof = self.extent_dof
        dof_excess = prior_dof - 2.0 * float(extent_dim) - 2.0
        predicted_dof = (
            extent_dim
            + 1.0
            + (prior_dof - extent_dim - 1.0) / (1.0 + dof_excess / transition_dof)
        )
        scale_factor = 1.0 / (
            1.0
            + (prior_dof - float(extent_dim) - 1.0)
            / (transition_dof - float(extent_dim) - 1.0)
        )
        self.extent_scale = self._project_symmetric_positive(
            scale_factor * A @ self.extent_scale @ A.T,
            self.minimum_extent_eigenvalue,
        )
        self.extent_dof = predicted_dof
        self.extent_transition_dof = transition_dof
        self.extent_transition_matrix = A

    # pylint: disable=too-many-locals
    def update(self, measurements, meas_mat, meas_noise_cov):
        if self.kinematic_state_to_pos_matrix is None:
            self.kinematic_state_to_pos_matrix = meas_mat

        ys = asarray(measurements)
        H = asarray(meas_mat)
        Cv = asarray(meas_noise_cov)

        y_rows, y_cols = ys.shape
        extent_dim = self.extent_dimension
        if y_rows != extent_dim:
            raise ValueError("measurement dimension must match extent dimension.")
        if y_cols < y_rows + 1:
            raise ValueError("Too few measurements.")

        y_mean = mean(ys, axis=1, keepdims=True)
        ys_demean = ys - y_mean
        measurement_scatter = ys_demean @ ys_demean.T

        prior_extent = self.extent
        predicted_measurement = H @ self.kinematic_state

        Y = self.measurement_spread_factor * prior_extent + Cv
        S = H @ self.covariance @ H.T + Y / y_cols
        K = self.covariance @ linalg.solve(S, H).T
        innovation = y_mean.flatten() - predicted_measurement
        self.kinematic_state = self.kinematic_state + K @ innovation
        self.covariance = self._symmetrize(self.covariance - K @ S @ K.T)

        Xsqrt = linalg.cholesky(prior_extent)
        Ssqrt = linalg.cholesky(S)
        Ysqrt = linalg.cholesky(Y)

        Nsqrt = Xsqrt @ linalg.solve(Ssqrt, innovation.reshape(-1, 1))
        innovation_extent = Nsqrt @ Nsqrt.T
        XYsqrt = linalg.solve(Ysqrt.T, Xsqrt.T).T
        scatter_extent = XYsqrt @ measurement_scatter @ XYsqrt.T

        self.extent_scale = self._project_symmetric_positive(
            self.extent_scale + innovation_extent + scatter_extent,
            self.minimum_extent_eigenvalue,
        )
        self.extent_dof = self.extent_dof + float(y_cols)

    def plot_point_estimate(self, scaling_factor=1, color=(0, 0.4470, 0.7410)):
        if self.kinematic_state_to_pos_matrix is None:
            raise ValueError("""No kinematic_state_to_pos_matrix
                             matrix was set, so it is unclear what
                             the individual components of the kinematic
                             state are (position, velocity, etc.).
                             Please set it directly or perform an update step
                             before plotting.""")
        position_estimate = self.kinematic_state_to_pos_matrix @ self.kinematic_state
        plot_ellipsoid(position_estimate, self.extent, scaling_factor, color)

    def get_contour_points(self, n):
        assert self.kinematic_state.shape == (2,)
        xs = linspace(0, 2 * pi, n)
        contour_points = self.kinematic_state[:, None] + self.extent @ array(
            [cos(xs), sin(xs)]
        )
        return contour_points.T


FactorizedGIWRMTracker = FactorizedGIWRandomMatrixTracker
