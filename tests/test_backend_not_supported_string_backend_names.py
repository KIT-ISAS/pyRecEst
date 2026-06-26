from pyrecest.exceptions import BackendNotSupportedError


def test_backend_not_supported_error_keeps_string_backend_name_whole():
    error = BackendNotSupportedError(
        "ExampleAPI.update",
        "jax",
        supported_backends="numpy",
    )

    assert error.supported_backends == ("numpy",)
    assert "supported backends: numpy" in str(error)
    assert "n, u, m" not in str(error)
