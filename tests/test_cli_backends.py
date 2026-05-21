import json

from pyrecest.cli import main


def test_backends_cli_json_output(capsys):
    assert main(["backends"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert "facade" in payload
    assert "api" in payload
    assert "KalmanFilter" in payload["api"]


def test_backends_cli_markdown_output(capsys):
    assert main(["backends", "--format", "markdown"]) == 0
    output = capsys.readouterr().out

    assert "| API | NumPy | PyTorch | JAX | Notes |" in output
    assert "`KalmanFilter`" in output
