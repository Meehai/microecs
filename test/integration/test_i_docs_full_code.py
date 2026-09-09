"""
test_i_docs_full_code.py - each example page ends with a `## Full code` block that must BE the example file.

The block is there so a reader can copy-paste one thing and have it run. That makes the file's source live in
two places, which is exactly the kind of duplicate that drifts silently -- so it is checked instead of trusted.
If this fails, the fix is to re-copy the .py into the .md (the file is the source of truth, never the other way).
"""
from pathlib import Path
import pytest

ROOT = Path(__file__).parents[2]
# pages live in docs/source/<Group>/; the menu path is derived from the directory tree.
PAGES = {"Examples & Tutorials/example-1-hello-world.md": "01-hello-world.py",
         "Examples & Tutorials/example-2-moving-colliding-balls.md": "02-moving-colliding-balls.py",
         "Examples & Tutorials/example-3-serialization.md": "03-serialization.py"}

def _full_code_block(md_text: str) -> str:
    """the python fenced block under the `## Full code` heading of a docs page"""
    _, _, after = md_text.partition("\n## Full code\n")
    assert after != "", "no '## Full code' section"
    _, _, after = after.partition("```python\n")
    assert after != "", "'## Full code' section has no ```python block"
    block, fence, _ = after.partition("\n```")
    assert fence != "", "unterminated ```python block"
    return block + "\n"

@pytest.mark.parametrize("page,example", PAGES.items(), ids=list(PAGES))
def test_full_code_block_is_the_example_file_verbatim(page: str, example: str):
    md_path, py_path = ROOT / "docs" / "source" / page, ROOT / "examples" / example
    assert md_path.exists() and py_path.exists(), (md_path, py_path)
    assert _full_code_block(md_path.read_text()) == py_path.read_text(), f"{page} drifted from {example}"

@pytest.mark.parametrize("page,example", PAGES.items(), ids=list(PAGES))
def test_page_links_to_the_example_file(page: str, example: str):
    """the page still points at the real file, so the block above is attributable"""
    assert f"examples/{example}" in (ROOT / "docs" / "source" / page).read_text()
