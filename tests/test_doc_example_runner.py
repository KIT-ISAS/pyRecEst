from pathlib import Path

from scripts.run_doc_examples import iter_python_blocks


def test_doc_example_runner_skips_marked_blocks(tmp_path: Path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        "```python\nprint('run')\n```\n\n```python\n# pyrecest: skip\nprint('skip')\n```\n",
        encoding="utf-8",
    )
    blocks = list(iter_python_blocks(doc))
    assert len(blocks) == 1
    assert "run" in blocks[0].code


def test_doc_example_runner_accepts_common_markdown_fence_variants(tmp_path: Path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        "```python title=\"demo\"\nprint('with attrs')\n```   \n\n"
        "   ```PY \r\nprint('indented uppercase crlf')\r\n   ```\r\n",
        encoding="utf-8",
    )

    blocks = list(iter_python_blocks(doc))

    assert [block.code for block in blocks] == [
        "print('with attrs')",
        "print('indented uppercase crlf')",
    ]
