from types import SimpleNamespace

import pytest
from pyrecest.backends import SUPPORTED_BACKENDS, get_backend
from pyrecest.exceptions import OptionalDependencyError


def test_get_backend_imports_supported_backend():
    backend = get_backend("numpy")
    assert backend.array([1.0]).shape == (1,)
    assert "numpy" in SUPPORTED_BACKENDS


def test_get_backend_accepts_autograd_backend(monkeypatch):
    imported = []
    fake_backend = SimpleNamespace(has_autodiff=lambda: True)

    def fake_import_module(module_name):
        imported.append(module_name)
        return fake_backend

    monkeypatch.setattr("pyrecest.backends.importlib.import_module", fake_import_module)

    assert "autograd" in SUPPORTED_BACKENDS
    assert get_backend("autograd") is fake_backend
    assert imported == ["pyrecest._backend.autograd"]


def test_get_backend_reports_optional_dependency_install_hint(monkeypatch):
    missing_dependency = ModuleNotFoundError("No module named 'torch'", name="torch")

    def fake_import_module(module_name):
        assert module_name == "pyrecest._backend.pytorch"
        raise missing_dependency

    monkeypatch.setattr("pyrecest.backends.importlib.import_module", fake_import_module)

    with pytest.raises(OptionalDependencyError) as exc_info:
        get_backend("pytorch")

    assert exc_info.value.__cause__ is missing_dependency
    message = str(exc_info.value)
    assert "Backend 'pytorch'" in message
    assert "torch" in message
    assert "pyrecest[pytorch_support]" in message


@pytest.mark.parametrize("name", [None, b"numpy", 1])
def test_get_backend_rejects_non_string_names(name):
    with pytest.raises(ValueError, match="Backend name must be a string"):
        get_backend(name)


def test_get_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("not-a-backend")
