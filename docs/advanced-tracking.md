# Advanced Tracking Notes

Advanced tracking workflows should make state, measurement-set, and association
semantics explicit before exposing a new example or public API.

Each advanced tracker example should document:

1. state vector layout and units;
2. measurement-set shape and missed-detection representation;
3. clutter model and gating configuration;
4. association output type and interpretation;
5. backend assumptions, especially NumPy/SciPy-only assignment paths;
6. evaluation metrics such as cardinality error, localization error, and track continuity.

Use small deterministic measurement sets for CI smoke tests. Assert stable
properties such as number of tracks, selected associations, finite log-likelihoods,
and diagnostic container shapes.
