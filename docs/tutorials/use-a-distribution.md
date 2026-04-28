# Use A Distribution

This tutorial creates Gaussian distributions, multiplies them, and checks the
result against the closed-form information representation.

Use `pyrecest.backend` arrays when you want code that can run on supported
numerical backends.

```python
from pyrecest.backend import array, diag, linalg, zeros
from pyrecest.distributions import GaussianDistribution


factors = [
    GaussianDistribution(array([0.0, 1.0]), diag(array([4.0, 1.0]))),
    GaussianDistribution(array([1.0, -0.5]), diag(array([1.0, 2.25]))),
    GaussianDistribution(array([-0.75, 0.25]), diag(array([0.5, 0.75]))),
]

product = factors[0]
for factor in factors[1:]:
    product = product.multiply(factor)

precision_sum = zeros(product.C.shape)
weighted_mean_sum = zeros(product.mu.shape)
for factor in factors:
    precision = linalg.inv(factor.C)
    precision_sum = precision_sum + precision
    weighted_mean_sum = weighted_mean_sum + precision @ factor.mu

reference_covariance = linalg.inv(precision_sum)
reference_mean = reference_covariance @ weighted_mean_sum

print("product mean:", product.mu)
print("reference mean:", reference_mean)
print("product covariance:")
print(product.C)
print("reference covariance:")
print(reference_covariance)
```

Run the same task as an executable example with:

```bash
python examples/basic/gaussian_multiplication.py
```

## What To Notice

- `GaussianDistribution(mu, C)` stores the mean in `mu` and covariance in `C`.
- `multiply()` returns a new distribution representing the normalized product.
- The information-form check adds precisions and precision-weighted means, which
  is a useful way to verify Gaussian products.
- Other distribution families follow the same broad pattern: construct the
  distribution, call methods such as `pdf`, `sample`, `multiply`, or
  `get_point_estimate`, and keep arrays backend-compatible where possible.
