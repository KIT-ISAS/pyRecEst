import importlib

import pyrecest.backend as backend


def test_backend_virtual_submodules_are_importable():
    for submodule_name in ("linalg", "random"):
        module = importlib.import_module(f"pyrecest.backend.{submodule_name}")

        assert module is getattr(backend, submodule_name)
