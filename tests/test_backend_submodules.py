import sys
from types import ModuleType

import pyrecest._backend_submodules as backend_submodules


def test_register_backend_submodules_refreshes_existing_entries(monkeypatch):
    first_backend = ModuleType("pyrecest.test_backend")
    first_backend.linalg = ModuleType("first_linalg")
    second_backend = ModuleType("pyrecest.test_backend")
    second_backend.linalg = ModuleType("second_linalg")
    module_name = "pyrecest.test_backend.linalg"

    monkeypatch.setattr(backend_submodules, "BACKEND_ATTRIBUTES", ("linalg",))
    try:
        backend_submodules.register_backend_submodules(first_backend)
        assert sys.modules[module_name] is first_backend.linalg

        backend_submodules.register_backend_submodules(second_backend)
        assert sys.modules[module_name] is second_backend.linalg
    finally:
        sys.modules.pop(module_name, None)
