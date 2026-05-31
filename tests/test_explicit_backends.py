from types import SimpleNamespace

import pytest
from pyrecest.backends import SUPPORTED_BACKENDS, get_backend


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


def test_get_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("not-a-backend")
