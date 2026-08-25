"""Headless smoke tests for the Streamlit UI (app.py + all pages).

Uses Streamlit's real AppTest harness (streamlit.testing.v1) to actually
execute each page's script top-to-bottom, not just check that it parses.
This module had zero test coverage before — the UI is the most visible
surface of the whole platform and nothing verified it even ran without
crashing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

UI_DIR = Path(__file__).resolve().parents[2] / "myoadapt" / "ui"
APP_FILE = UI_DIR / "app.py"
PAGE_FILES = sorted((UI_DIR / "pages").glob("*.py"))


def test_app_py_exists_and_pages_exist():
    assert APP_FILE.exists()
    assert len(PAGE_FILES) == 5


def test_landing_page_runs_without_exception():
    at = AppTest.from_file(str(APP_FILE))
    at.run(timeout=30)
    assert not at.exception, f"app.py raised: {at.exception}"
    assert len(at.markdown) > 0


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.name)
def test_page_runs_without_exception(page):
    at = AppTest.from_file(str(page))
    at.run(timeout=30)
    assert not at.exception, f"{page.name} raised: {at.exception}"


def test_landing_page_kpi_values_are_not_placeholder_duplicates():
    """Regression test for a real bug: every KPI card showed the same
    hardcoded '179' value instead of its own metric."""
    at = AppTest.from_file(str(APP_FILE))
    at.run(timeout=30)
    full_html = "\n".join(md.value for md in at.markdown if md.value)
    assert full_html.count(">179<") == 0


def test_landing_page_paper_cards_have_no_indented_html_lines():
    """Regression test for a real bug found via actual screenshot
    verification, not catchable by AppTest's exception-free check alone:
    the paper-chain cards were built via repeated += on an indented
    multi-line f-string. Streamlit's markdown renderer treated the
    leading whitespace on each appended line as an indented code block,
    so every card after the first rendered as literal visible text
    ('<div class="ma-paper-card">...') instead of an actual element.

    AppTest inspects the markdown *source string*, not a real rendered
    DOM — so checking whether card title text is merely present doesn't
    distinguish the bug from the fix (both contain the same text; only
    the browser's interpretation of it differs). The meaningful static
    check is for the dangerous pattern itself: a newline followed by
    leading whitespace then a '<div' tag. Scoped to just the paper-chain
    markdown block specifically (identified by its container class) —
    other elements on the page (hero, spacer, info_card) are legitimate
    single self-contained st.markdown() calls that also contain
    multi-line indented HTML in their source but render correctly; the
    bug was specific to concatenating multiple blocks via += in a loop,
    not indentation in isolation, so a page-wide scan would false-flag
    those.
    """
    at = AppTest.from_file(str(APP_FILE))
    at.run(timeout=30)
    paper_blocks = [md.value for md in at.markdown
                    if md.value and '<div class="ma-paper-chain">' in md.value]
    assert len(paper_blocks) == 1, "expected exactly one paper-chain markdown block"
    paper_html = paper_blocks[0]
    assert paper_html.count('class="ma-paper-card"') == 5
    dangerous_pattern = re.compile(r"\n[ \t]+<div")
    match = dangerous_pattern.search(paper_html)
    assert match is None, (
        f"Found an indented '<div' after a newline in the paper-chain "
        f"block near: {paper_html[max(0, match.start() - 40):match.start() + 40]!r} "
        f"— this is the exact pattern that broke paper-card rendering."
    )
