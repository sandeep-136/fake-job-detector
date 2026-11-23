import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prepare import build_text, deduplicate


def _row(**kw):
    base = {c: "" for c in ["title", "company_profile", "description", "requirements", "benefits"]}
    base.update(kw)
    return pd.Series(base)


def test_build_text_joins_present_fields():
    text = build_text(_row(title="Engineer", description="Build things.", benefits=""))
    assert text == "Engineer\nBuild things."


def test_build_text_skips_nan():
    row = _row(title="Engineer", description=float("nan"))
    assert build_text(row) == "Engineer"


def test_deduplicate_removes_exact_duplicates():
    rows = [
        {"title": "A", "description": "x", "fraudulent": 1},
        {"title": "A", "description": "x", "fraudulent": 1},
        {"title": "B", "description": "y", "fraudulent": 0},
    ]
    df = pd.DataFrame(rows)
    out = deduplicate(df)
    assert len(out) == 2
    assert set(out["title"]) == {"A", "B"}


def test_deduplicate_keeps_same_title_different_body():
    rows = [
        {"title": "A", "description": "x"},
        {"title": "A", "description": "different"},
    ]
    assert len(deduplicate(pd.DataFrame(rows))) == 2
