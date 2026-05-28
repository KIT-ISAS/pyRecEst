import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_jax_euclidean_ukf_predict_update_runs_without_inplace_writes():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "jax"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter
from pyrecest.sampling.sigma_points import MerweScaledSigmaPoints

points = MerweScaledSigmaPoints(2, alpha=0.5, beta=2.0, kappa=0.0)
sigmas = points.sigma_points(backend.asarray([0.0, 0.0]), backend.eye(2))
assert sigmas.shape == (5, 2)
assert bool(backend.to_numpy(backend.isclose(backend.sum(points.Wm), 1.0)))

initial = GaussianDistribution(backend.asarray([0.0, 0.0]), backend.eye(2))
ukf = UnscentedKalmanFilter(initial)
ukf.predict_identity(0.1 * backend.eye(2))
ukf.update_identity(0.1 * backend.eye(2), backend.asarray([1.0, -1.0]))
estimate = ukf.get_point_estimate()
assert estimate.shape == (2,)
assert bool(backend.to_numpy(backend.all(backend.isfinite(estimate))))
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
