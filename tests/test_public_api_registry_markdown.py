from scripts import check_public_api_registry as registry_script


def test_public_api_registry_markdown_cell_replaces_carriage_return():
    rendered = registry_script._markdown_table_cell("first\rsecond")

    assert rendered == "first second"
