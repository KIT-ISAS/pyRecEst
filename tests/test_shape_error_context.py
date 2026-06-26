from pyrecest.exceptions import ShapeError


def test_shape_error_message_includes_context_without_actual_shape():
    error = ShapeError(
        "measurement",
        expected="shape (n,)",
        reason="measurement vector missing",
    )

    assert "measurement" in str(error)
    assert "shape (n,)" in str(error)
    assert "measurement vector missing" in str(error)
