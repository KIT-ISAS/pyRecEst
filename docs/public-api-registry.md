# Public API Registry

The public API registry is the set of names exported by package-level namespaces
such as `pyrecest.filters`. It complements the backend API matrix: the registry
answers "what name is public?" and the matrix answers "which backend supports
that name?"

## Rules

1. Prefer one canonical spelling for each public object.
2. Keep compatibility aliases only when they protect existing user code or a
   documented external naming convention.
3. Add every new backend-sensitive object to `docs/backend-api-matrix.md` and
   `src/pyrecest/_backend/capabilities.py`.
4. Keep lazy export maps, `__all__`, and documentation synchronized with tests.

## Filters Namespace

`pyrecest.filters` is lazy: importing the namespace does not import every
tracker implementation. Add new filter symbols to `_FILTER_EXPORTS` in
`src/pyrecest/filters/__init__.py`; `__all__` is generated from that map.

Compatibility aliases such as mixed acronym/camelcase tracker names should stay
mapped to the same implementation module as their canonical form. New examples
and documentation should use the canonical spelling.
