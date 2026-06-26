import pyrecest


def test_is_backend_accepts_expected_name_tuples():
    active = pyrecest.get_backend_name()
    other = "jax" if active != "jax" else "numpy"

    assert pyrecest.is_backend((other, active))
