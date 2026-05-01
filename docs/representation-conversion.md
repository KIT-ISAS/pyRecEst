# Representation Conversion

PyRecEst distributions often support several representations of the same
uncertainty state: analytic densities, Dirac/particle sets, grids, Fourier
series, mixtures, and moment-matched approximations.

Use `convert_distribution` or the method alias `approximate_as` to make these
conversions explicit and discoverable.

```python
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution
from pyrecest.distributions.conversion import convert_distribution

prior = GaussianDistribution(array([0.0, 0.0]), eye(2))
particles = convert_distribution(prior, LinearDiracDistribution, n_particles=1000)
```

The same conversion can be written from the source distribution when the source
inherits from one of the common PyRecEst distribution base classes:

```python
particles = prior.approximate_as(LinearDiracDistribution, n_particles=1000)
```

## Target-centric conversions

Conversions are target-centric. A target class can expose
`from_distribution(distribution, ...)`, and the generic conversion gateway will
call it.

This keeps domain-specific approximation logic close to the representation that
owns it. For example:

- `LinearDiracDistribution.from_distribution(...)` samples a source density.
- `CircularGridDistribution.from_distribution(...)` evaluates a circular density
  on a grid.
- `CircularFourierDistribution.from_distribution(...)` builds Fourier
  coefficients from samples or grid values.
- `GaussianDistribution.from_distribution(...)` performs Gaussian moment
  matching when the source exposes `mean()` and `covariance()`.

## Metadata

Use `return_info=True` when you need to know how the conversion was performed.

```python
result = convert_distribution(
    prior,
    LinearDiracDistribution,
    n_particles=1000,
    return_info=True,
)

particles = result.distribution
print(result.method)
print(result.exact)
```

Identity conversions are exact. Sampling, grid approximation, Fourier
truncation, and moment matching are reported as approximate unless a converter
explicitly marks them as exact.

## Custom conversions

Third-party representations can register conversions without editing central
dispatch code.

```python
from pyrecest.distributions.conversion import register_conversion


@register_conversion(MyDistribution, MyParticleDistribution)
def my_distribution_to_particles(distribution, n_particles):
    return MyParticleDistribution(distribution.sample(n_particles))
```

After registration, the normal gateway works:

```python
particles = convert_distribution(source, MyParticleDistribution, n_particles=1000)
```

## Common parameters

Common conversion parameters depend on the target representation:

- `n_particles` for Dirac/particle approximations.
- `no_of_gridpoints` for one-dimensional circular grids.
- `n_grid_points` for hypertoroidal grids.
- `no_of_grid_points` and `grid_type` for hypersphere-subset grids.
- `n` for Fourier representations.
