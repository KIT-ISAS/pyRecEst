# Public Protocols

`pyrecest.protocols` is the public home for small capability contracts used by
PyRecEst components.

A protocol describes what an object can do without requiring it to inherit from
a specific base class. This keeps extension points lightweight: a user-defined
class can work with generic PyRecEst utilities when it implements the required
methods and attributes.

## Current scope

This seed package only defines common dimension protocols and broad array
aliases:

- `SupportsDim` for objects with an intrinsic state-space dimension;
- `SupportsInputDim` for objects with an ambient or input coordinate dimension;
- `ArrayLike` and `BackendArray` as intentionally broad aliases for backend
  compatible values.

Follow-up pull requests can add distribution, filter, model, conversion, and
manifold-specific protocols independently.

## Design principles

Protocols should stay small and capability-oriented. Instead of requiring every
distribution, model, or filter to implement one large interface, PyRecEst should
ask only for the capability that a function actually needs.

For example, a future density utility may require a `SupportsPdf` protocol while
a sampler utility may require only `SupportsSampling`. A particle representation
should not need to implement analytic density evaluation merely to satisfy a
large distribution base interface.

Prefer additive protocols over mandatory base classes:

```python
from pyrecest.protocols.common import SupportsDim


def describe_dimension(obj: SupportsDim) -> str:
    return f"intrinsic dimension: {obj.dim}"
```

A class does not need to inherit from `SupportsDim`; it only needs to expose the
required attribute or property.
For filters, the same principle means a generic history utility can require only
`SupportsHistoryRecording`, while a linear-Gaussian benchmark can require
`LinearFilterLike`. A particle, grid, nonlinear, or tracker-style filter should
not need to implement linear prediction merely to satisfy a large filter base
interface.

## Filter capability protocols

Filter protocols live in `pyrecest.protocols.filters` and follow existing
PyRecEst method names such as `filter_state`, `get_point_estimate`,
`predict_linear`, `update_linear`, `predict_nonlinear`, `update_nonlinear`,
`predict_model`, and `update_model`.

Use small protocols when a function needs only one capability:

```python
from pyrecest.protocols.filters import SupportsPointEstimate


def read_estimate(filter_: SupportsPointEstimate):
    return filter_.get_point_estimate()
```

Use composed protocols when a function needs a complete filter style:

```python
from pyrecest.protocols.filters import LinearFilterLike


def run_linear_step(
    filter_: LinearFilterLike,
    f_matrix,
    q_matrix,
    z,
    h_matrix,
    r_matrix,
):
    filter_.predict_linear(f_matrix, q_matrix)
    filter_.update_linear(z, h_matrix, r_matrix)
    return filter_.get_point_estimate()
```

The identity-prediction and identity-update protocols intentionally expose only
the shared structural method names. Existing filters use those method names with
slightly different optional argument semantics, so code that needs stronger
semantics should prefer `SupportsLinearPredict`, `SupportsLinearUpdate`, or a
model-based protocol.

## Runtime checks

The public protocols are runtime-checkable where practical:

```python
from pyrecest.protocols.common import SupportsDim


class DemoObject:
    dim = 2


assert isinstance(DemoObject(), SupportsDim)
```

Runtime checks confirm that the required attributes or methods are present. They
do not prove mathematical correctness. Protocol-specific tests should check
shapes, backend behavior, and semantics separately.

## Import style

Use submodule imports in early protocol pull requests:

```python
from pyrecest.protocols.common import SupportsDim, SupportsInputDim
```

Package-level exports are intentionally minimal in this seed package to reduce
merge conflicts while follow-up protocol modules are developed in parallel.

## Extension examples

The extension guides show how to write user-defined components that follow the
current public protocol seed and existing PyRecEst naming conventions:

- [Custom distribution extensions](custom-distribution.md) explains how to build
  a small distribution-like class with `dim`, `input_dim`, `pdf`, `sample`,
  `mean`, and `covariance` methods.
- [Custom filter extensions](custom-filter.md) explains how to build a small
  recursive filter-like class with `dim`, `filter_state`, prediction, update,
  and point-estimate methods.

The runnable scripts are:

```bash
python examples/basic/custom_distribution_protocol.py
python examples/basic/custom_filter_protocol.py
```

These examples intentionally depend only on the currently available common
protocols. When distribution- and filter-specific protocol modules are added,
the same examples can be updated to import those narrower capability protocols.
