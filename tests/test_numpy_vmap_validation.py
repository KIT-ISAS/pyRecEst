import numpy as np
import pytest
from pyrecest._backend import numpy as numpy_backend


def test_numpy_vmap_rejects_scalar_arguments():
    mapped = numpy_backend.vmap(lambda x: x)

    with pytest.raises(ValueError, match="at least one dimension"):
        mapped(np.array(1.0))


def test_numpy_vmap_rejects_mixed_scalar_arguments():
    mapped = numpy_backend.vmap(lambda x, y: x + y)

    with pytest.raises(ValueError, match="at least one dimension"):
        mapped(np.array([1.0]), np.array(2.0))
