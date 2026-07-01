import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_logistic_pairwise_standardization_uses_backend_std_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np

import pyrecest.backend as backend
from pyrecest.utils import LogisticPairwiseAssociationModel

features_np = np.asarray(
    [
        [1.0, 2.0],
        [3.0, 8.0],
        [5.0, 14.0],
        [7.0, 20.0],
    ],
    dtype=float,
)
features = backend.asarray(features_np, dtype=backend.float64)
model = LogisticPairwiseAssociationModel()
model._fit_standardization(features)

actual_scale = backend.to_numpy(model.feature_scale_)
expected_population_scale = features_np.std(axis=0)
unbiased_torch_scale = features_np.std(axis=0, ddof=1)

np.testing.assert_allclose(actual_scale, expected_population_scale, rtol=1e-12, atol=1e-12)
assert not np.allclose(actual_scale, unbiased_torch_scale)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
