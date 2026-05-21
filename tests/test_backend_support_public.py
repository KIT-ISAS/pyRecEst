from pyrecest import backend_support, format_backend_support_markdown, get_backend_support
from pyrecest._backend.capabilities import BACKEND_SUPPORT_LEVELS


def test_public_backend_support_lookup():
    assert get_backend_support("KalmanFilter", backend="numpy") == "supported"
    assert backend_support("EvaluationUtilities", backend="pytorch") in BACKEND_SUPPORT_LEVELS
    assert get_backend_support("missing-api") is None


def test_backend_support_markdown_contains_expected_rows():
    rendered = format_backend_support_markdown()
    assert "KalmanFilter" in rendered
    assert "BackendFacade" in rendered
