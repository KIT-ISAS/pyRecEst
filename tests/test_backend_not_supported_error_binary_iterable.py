from pyrecest.exceptions import BackendNotSupportedError


def test_backend_not_supported_error_decodes_binary_supported_backend_iterables():
    error = BackendNotSupportedError(
        "ExampleAPI.update",
        "jax",
        supported_backends=(
            bytes("numpy", "utf-8"),
            bytearray("pytorch", "utf-8"),
        ),
    )

    assert error.supported_backends == ("numpy", "pytorch")
    assert str(error) == (
        "ExampleAPI.update is unavailable for backend 'jax'; "
        "supported backends: numpy, pytorch"
    )
