import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import mod, pi

from .abstract_circular_distribution import AbstractCircularDistribution


class PiecewiseConstantDistribution(AbstractCircularDistribution):
    """Piecewise constant (i.e. discrete) circular distribution, similar to a histogram.

    The circle [0, 2*pi) is divided into n equal intervals, each with a constant
    probability density weight.

    Gerhard Kurz, Florian Pfaff, Uwe D. Hanebeck,
    Discrete Recursive Bayesian Filtering on Intervals and the Unit Circle
    Proceedings of the 2016 IEEE International Conference on Multisensor Fusion
    and Integration for Intelligent Systems (MFI 2016),
    Baden-Baden, Germany, September 2016.
    """

    def __init__(self, w):
        """Initialize with a weight vector that is automatically normalized.

        Parameters
        ----------
        w : array_like, shape (n,)
            Weight for each interval (will be normalized to form a valid pdf).
        """
        AbstractCircularDistribution.__init__(self)
        w = np.asarray(w, dtype=float).ravel()
        assert w.ndim == 1 and w.size > 0
        self.w = w / (np.mean(w) * 2.0 * np.pi)

    def pdf(self, xs):
        """Evaluate the pdf at each point in xs.

        Parameters
        ----------
        xs : array_like, shape (n,)
            Points at which to evaluate the pdf.

        Returns
        -------
        p : ndarray, shape (n,)
            Pdf values at each point.
        """
        assert xs.ndim == 1
        xs_mod = np.asarray(mod(xs, 2.0 * pi), dtype=float)
        n = len(self.w)
        idx = np.minimum(
            np.floor(xs_mod / (2.0 * np.pi) * n).astype(int), n - 1
        )
        return self.w[idx]

    def trigonometric_moment(self, n):
        """Calculate the n-th trigonometric moment analytically.

        Parameters
        ----------
        n : int
            Moment order.

        Returns
        -------
        m : complex
            n-th trigonometric moment.
        """
        if n == 0:
            return 1.0 + 0j
        num = len(self.w)
        interv = np.zeros(num, dtype=complex)
        for j in range(1, num + 1):
            l = PiecewiseConstantDistribution.left_border(j, num)
            r = PiecewiseConstantDistribution.right_border(j, num)
            c = PiecewiseConstantDistribution.interval_center(j, num)
            w_j = float(self.pdf(np.array([c]))[0])
            interv[j - 1] = w_j * (np.exp(1j * n * r) - np.exp(1j * n * l))
        return complex(-1j / n * np.sum(interv))

    def entropy(self):
        """Calculate the entropy analytically.

        Returns
        -------
        e : float
            Entropy of the distribution.
        """
        n = len(self.w)
        return float(-2.0 * np.pi / n * np.sum(self.w * np.log(self.w)))

    def sample(self, n):
        """Draw n random samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : ndarray, shape (n,)
            Samples in [0, 2*pi).
        """
        num_intervals = len(self.w)
        interval_width = 2.0 * np.pi / num_intervals
        # Each interval has probability w[j] * interval_width, which sums to 1 by
        # construction. Divide by sum anyway to guard against floating-point drift.
        interval_probs = self.w * interval_width
        interval_probs /= interval_probs.sum()
        interval_indices = np.random.choice(num_intervals, size=n, p=interval_probs)
        return interval_indices * interval_width + np.random.uniform(
            0.0, interval_width, size=n
        )

    @staticmethod
    def left_border(m, n):
        """Left border of the m-th interval (1-indexed) for n total intervals.

        Parameters
        ----------
        m : int
            Interval index (1-indexed).
        n : int
            Total number of intervals.

        Returns
        -------
        float
            Left border of the m-th interval.
        """
        assert 1 <= m <= n
        return 2.0 * np.pi / n * (m - 1)

    @staticmethod
    def right_border(m, n):
        """Right border of the m-th interval (1-indexed) for n total intervals.

        Parameters
        ----------
        m : int
            Interval index (1-indexed).
        n : int
            Total number of intervals.

        Returns
        -------
        float
            Right border of the m-th interval.
        """
        assert 1 <= m <= n
        return 2.0 * np.pi / n * m

    @staticmethod
    def interval_center(m, n):
        """Center of the m-th interval (1-indexed) for n total intervals.

        Parameters
        ----------
        m : int
            Interval index (1-indexed).
        n : int
            Total number of intervals.

        Returns
        -------
        float
            Center of the m-th interval.
        """
        assert 1 <= m <= n
        return 2.0 * np.pi / n * (m - 0.5)

    @staticmethod
    def calculate_parameters_numerically(pdf_func, n):
        """Calculate weights by numerically integrating a given pdf over each interval.

        Parameters
        ----------
        pdf_func : callable
            Pdf of a circular density; accepts a 1-D array and returns a 1-D array.
        n : int
            Number of discretization intervals.

        Returns
        -------
        w : ndarray, shape (n,)
            Weights of the corresponding PiecewiseConstantDistribution.
        """
        from scipy.integrate import quad  # pylint: disable=import-outside-toplevel

        assert n >= 1
        w = np.zeros(n)
        for j in range(1, n + 1):
            l = PiecewiseConstantDistribution.left_border(j, n)
            r = PiecewiseConstantDistribution.right_border(j, n)
            w[j - 1] = quad(
                lambda x: float(pdf_func(np.array([x]))), l, r
            )[0]
        return w
