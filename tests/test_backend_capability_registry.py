from pyrecest._backend import capabilities
from pyrecest._backend.capabilities import validate_api_backend_capabilities


def test_api_backend_capability_registry_is_well_formed():
    errors = validate_api_backend_capabilities()
    assert errors == (), "\n".join(errors)


def test_api_backend_capability_registry_rejects_unknown_keys(monkeypatch):
    broken_registry = dict(capabilities.API_BACKEND_CAPABILITIES)
    broken_registry["BrokenAPI"] = {
        "numpy": "supported",
        "pytorch": "supported",
        "jax": "supported",
        "tensorflow": "supported",
        "notes": "Synthetic row with a misspelled backend key.",
    }

    monkeypatch.setattr(capabilities, "API_BACKEND_CAPABILITIES", broken_registry)

    errors = capabilities.validate_api_backend_capabilities()

    assert "BrokenAPI: unknown capability entries for tensorflow." in errors
