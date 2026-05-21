from scripts.generate_release_notes import render_release_notes


def test_release_notes_grouping():
    notes = render_release_notes(["feat: add tracker", "fix: handle shape", "unprefixed change"])
    assert "## Features" in notes
    assert "## Fixes" in notes
    assert "## Other changes" in notes
