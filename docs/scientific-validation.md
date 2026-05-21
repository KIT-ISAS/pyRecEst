# Scientific Validation

Scientific regressions are often subtler than import or shape failures. The unit
suite should include mathematical invariants that are stable across small
implementation changes.

| Component | Invariant |
|-----------|-----------|
| Probability distributions | Densities integrate or sum to one on their manifold. |
| Circular and toroidal distributions | Wrapping by the period preserves density. |
| Hyperspherical distributions | Samples and support points remain unit norm. |
| Gaussian filters | Covariances stay symmetric positive definite after updates. |
| Particle filters | Weights remain finite, non-negative, and normalized after resampling. |
| Representation conversion | Moment-matching routes preserve mean/covariance within tolerance. |
| Trackers | Cardinality, association, and gating diagnostics remain internally consistent. |

Prefer invariant checks over brittle full-trajectory comparisons unless the
scenario is intentionally deterministic and has a documented golden output.
