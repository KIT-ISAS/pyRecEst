from pyrecest.exceptions import BackendNotSupportedError


def test_backend_not_supported_error_coerces_backend_names_to_strings():
    error = BackendNotSupportedError(
        "ExampleAPI.update",
        "jax",
        supported_backends=("numpy", 123),
    )

    assert error.supported_backends == ("numpy", "123")
    assert "supported backends: numpy, 123" in str(error)
