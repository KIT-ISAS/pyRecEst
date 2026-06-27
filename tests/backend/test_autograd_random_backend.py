import pytest

pytest.importorskip("autograd")

from pyrecest._backend.autograd import random  # noqa: E402


def test_autograd_random_backend_imports():
    assert random is not None
