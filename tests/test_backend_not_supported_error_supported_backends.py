from pyrecest.exceptions import BackendNotSupportedError


def test_backend_not_supported_error_treats_supported_backends_string_as_one_name():
    error = BackendNotSupportedError(
        "ExampleAPI.update",
        "jax",
        supported_backends="numpy",
    )

    assert error.supported_backends == ("numpy",)
    assert str(error).endswith("supported backends: numpy")
