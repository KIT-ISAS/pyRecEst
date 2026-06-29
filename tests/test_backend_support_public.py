import importlib.util
from importlib import import_module

import pytest
from tests.support.backend_runner import run_backend_code

from pyrecest import (
    backend_support,
    format_backend_support_markdown,
    get_backend_support,
)
from pyrecest._backend.capabilities import BACKEND_SUPPORT_LEVELS

backend_support_module = import_module("pyrecest.backend_support")


def test_public_backend_support_lookup():
    assert get_backend_support("KalmanFilter", backend="numpy") == "supported"
    assert (
        backend_support("EvaluationUtilities", backend="pytorch")
        in BACKEND_SUPPORT_LEVELS
    )
    assert get_backend_support("missing-api") is None


def test_backend_support_markdown_contains_expected_rows():
    rendered = format_backend_support_markdown()
    assert "KalmanFilter" in rendered
    assert "BackendFacade" in rendered


def test_backend_support_markdown_preserves_table_separators(monkeypatch):
    separator = chr(124)
    escaped_separator = chr(92) + separator

    def fake_backend_capabilities():
        return (
            (
                f"Pipe{separator}API",
                {
                    "numpy": "supported",
                    "pytorch": f"partial{separator}bridged",
                    "jax": "unsupported",
                    "notes": f"first {separator} second\ncontinued",
                },
            ),
        )

    monkeypatch.setattr(
        backend_support_module,
        "iter_api_backend_capabilities",
        fake_backend_capabilities,
    )

    rendered = backend_support_module.format_backend_support_markdown()
    data_row = rendered.splitlines()[-1]

    assert data_row.count(separator) == 9
    assert f"Pipe{escaped_separator}API" in data_row
    assert f"partial{escaped_separator}bridged" in data_row
    assert f"first {escaped_separator} second<br>continued" in data_row


@pytest.mark.backend_portable
def test_pytorch_pad_edge_mode_matches_numpy_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

values = backend.array([[1, 2, 3], [4, 5, 6]])
padded = backend.pad(values, ((1, 1), (2, 1)), mode="edge")

expected = np.pad(
    np.array([[1, 2, 3], [4, 5, 6]]),
    ((1, 1), (2, 1)),
    mode="edge",
)
assert backend.to_numpy(padded).tolist() == expected.tolist()
""",
    )

    assert result.returncode == 0, result.stderr
