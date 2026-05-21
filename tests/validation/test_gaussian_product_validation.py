import pytest
from examples.basic.gaussian_multiplication import run_example
from pyrecest.backend import allclose


@pytest.mark.validation
def test_gaussian_product_matches_closed_form_information_result():
    _factors, product, reference_product = run_example()
    assert bool(allclose(product.mu, reference_product.mu))
    assert bool(allclose(product.C, reference_product.C))
