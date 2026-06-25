import importlib

from pyrecest import (
    backend_support,
    format_backend_support_markdown,
    get_backend_support,
)
from pyrecest._backend.capabilities import BACKEND_SUPPORT_LEVELS


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


def test_backend_support_markdown_escapes_table_cells(monkeypatch):
    backend_support_module = importlib.import_module("pyrecest.backend_support")
    monkeypatch.setattr(
        backend_support_module,
        "iter_api_backend_capabilities",
        lambda: (
            (
                "Escaped|API",
                {
                    "numpy": "supported|native",
                    "pytorch": "partial",
                    "jax": "unsupported",
                    "notes": "contains | pipe\nand newline",
                },
            ),
        ),
    )

    rendered = backend_support_module.format_backend_support_markdown()
    rows = rendered.splitlines()

    assert len(rows) == 3
    assert r"`Escaped\|API`" in rows[2]
    assert r"supported\|native" in rows[2]
    assert r"contains \| pipe<br>and newline" in rows[2]
