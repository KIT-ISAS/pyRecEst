from pyrecest._backend.capabilities import validate_api_backend_capabilities


def test_api_backend_capability_registry_is_well_formed():
    errors = validate_api_backend_capabilities()
    assert errors == (), "\n".join(errors)
