import numpy as np
import pytest
from pyrecest.filters import linear_update_planning as lup


class Measurement:
    source = "s"
    vector = np.array([0.0])


def test_lookup_rejects_bool_scalar_value():
    with pytest.raises(ValueError, match="finite scalar"):
        lup.source_float_value(Measurement(), {"s": True}, default=0.0)
