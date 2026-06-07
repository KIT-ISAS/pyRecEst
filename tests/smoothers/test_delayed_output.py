from __future__ import annotations

from pyrecest.smoothers import DelayedStateOutputMixin


class _DummyDelayedOutputSmoother(DelayedStateOutputMixin):
    def __init__(self):
        self._initialize_delayed_state_outputs()


def test_delayed_output_queue_returns_items_once() -> None:
    smoother = _DummyDelayedOutputSmoother()

    assert smoother._queue_delayed_state(0, "zero")
    assert smoother._queue_delayed_state(1, "one")
    assert not smoother._queue_delayed_state(1, "duplicate")

    assert smoother.pop_ready_states() == [(0, "zero"), (1, "one")]
    assert smoother.pop_ready_states() == []
    assert smoother.last_emitted_step == 1


def test_finalize_flushes_queued_states_and_unemitted_tail() -> None:
    smoother = _DummyDelayedOutputSmoother()
    smoother._queue_delayed_state(2, "queued")

    finalized = smoother._finalize_delayed_state_outputs(
        4,
        lambda step: f"tail-{step}",
    )

    assert finalized == [(2, "queued"), (3, "tail-3"), (4, "tail-4")]
    assert smoother.pop_ready_states() == []
    assert smoother.last_emitted_step == 4


def test_finalize_allows_missing_tail_states() -> None:
    smoother = _DummyDelayedOutputSmoother()

    finalized = smoother._finalize_delayed_state_outputs(
        2,
        lambda step: None if step == 1 else f"state-{step}",
    )

    assert finalized == [(0, "state-0"), (2, "state-2")]
    assert smoother.last_emitted_step == 2


def test_delayed_output_mixin_initializes_lazily() -> None:
    smoother = DelayedStateOutputMixin()

    assert smoother.pop_ready_states() == []
    assert smoother._queue_delayed_state(0, "state")
    assert smoother.pop_ready_states() == [(0, "state")]
    assert smoother.last_emitted_step == 0
