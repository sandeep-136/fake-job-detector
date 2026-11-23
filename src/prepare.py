"""Data preparation for the fake job posting detector.

Shared by training (M1) and the inference service (M2) so that the text a
model sees at prediction time is built exactly the way it was at training
time.
"""

import pandas as pd

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
META_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]
LABEL_COLUMN = "fraudulent"


def build_text(row: pd.Series) -> str:
    """Concatenate the free-text fields of one posting into a single string."""
    parts = []
    for col in TEXT_COLUMNS:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def load_postings(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and add a combined `text` column. `job_id` is dropped:
    it is an enumeration artifact, not a property of the posting."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["job_id"], errors="ignore")
    df["text"] = df.apply(build_text, axis=1)
    return df


def _normalize(text: str) -> str:
    """Lowercase, collapse non-alphanumerics: catches duplicates that differ
    only in case, punctuation or whitespace."""
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())).strip()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate postings before any split.

    Two passes: exact duplicates (identical text fields), then duplicates of
    the normalized combined text (differing only in case/punctuation/spacing).
    Left in, copies of the same posting can land on both sides of a random
    split and inflate evaluation scores.

    This does NOT catch reworded or templated near-duplicates; train.py
    quantifies those separately instead of claiming they are handled.
    """
    before = len(df)
    subset = [c for c in TEXT_COLUMNS if c in df.columns]
    df = df.drop_duplicates(subset=subset, keep="first")
    exact_removed = before - len(df)
    df["_norm"] = df["text"].map(_normalize) if "text" in df.columns else df[subset].fillna("").agg(" ".join, axis=1).map(_normalize)
    df = df.drop_duplicates(subset=["_norm"], keep="first").drop(columns=["_norm"]).reset_index(drop=True)
    norm_removed = before - exact_removed - len(df)
    print(f"deduplicate: removed {exact_removed} exact + {norm_removed} case/punctuation duplicates, {len(df)} postings remain")
    return df


def train_test_frame(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split on the deduplicated frame."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[LABEL_COLUMN]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
