from pyrecest.utils.track_completion import enumerate_fragment_completion_paths


def test_text_candidate_observation_is_rejected():
    candidate_text = str(1)
    provider = lambda *_: [candidate_text]

    try:
        enumerate_fragment_completion_paths(
            [[0, None]], direction="suffix", candidate_provider=provider
        )
    except ValueError as exc:
        assert "candidate observations must be non-negative integers" in str(exc)
    else:
        raise AssertionError("text-valued candidate observation was accepted")
