# Change Management

Broad improvements are easier to review and debug when split into independent
change sets. Prefer a series of focused PRs over one all-encompassing PR.

## Review Checklist

- The public API and stability category are clear.
- Backend behavior is declared as supported, bridged, partial, or unsupported.
- Numerical validation policy is explicit.
- User-facing examples or docs were updated.
- A small regression or scenario test protects the behavior.
- Release notes can describe the user-visible change in one sentence.

When a change touches multiple rows in this checklist, consider splitting it.
