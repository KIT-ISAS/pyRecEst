import jax.numpy as jnp
from jax import vmap
from jax.numpy import sin, cos

def _hypersph_to_cart_colatitude(r, *angles):
    """
    Convert hyperspherical coordinates to Cartesian coordinates in n-dimensions using JAX.
    Handles vector-valued inputs for angles.

    Parameters:
    - r (float or array): The radial distance(s).
    - angles (array): Arrays of angles, each with the same number of rows.

    Returns:
    - array: Cartesian coordinates for each set of input angles.
    """
    if len(angles) == 0:
        return jnp.atleast_2d(r).T

    r = jnp.atleast_1d(r)
    if r.ndim == 1:
        r = r[:, jnp.newaxis]

    # Create a matrix of sine values
    sin_matrix = jnp.array([sin(angle) for angle in angles])
    sin_matrix = jnp.tril(sin_matrix.T).T
    sin_matrix = jnp.where(sin_matrix == 0, 1, sin_matrix)

    # Compute the product over rows for sine values
    sin_product = jnp.prod(sin_matrix, axis=0)

    # Cosine values for Cartesian coordinates
    cos_values = jnp.array([cos(angle) for angle in angles] + [jnp.ones_like(angles[0])])

    # Compute Cartesian coordinates
    coords = r * sin_product * cos_values

    return coords.T

# Example usage
import jax
r = 1
theta1 = jax.random.uniform(jax.random.PRNGKey(0), (10,)) * jnp.pi
theta2 = jax.random.uniform(jax.random.PRNGKey(1), (10,)) * 2 * jnp.pi
coordinates = _hypersph_to_cart_colatitude(r, theta1, theta2)
print(coordinates)
