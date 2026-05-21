from pyrecest.filters.association_hypotheses import association_backend_support


def test_association_backend_boundary_is_explicit():
    support = association_backend_support()

    assert "active_backend" in support
    assert "support" in support
    assert support["support"] in {"native", "numpy_scipy_boundary"}
