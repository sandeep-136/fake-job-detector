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


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate postings (identical text fields).

    The public dataset contains many exact duplicates. If duplicates are left
    in, copies of the same posting can land on both sides of a random split
    and inflate evaluation scores, so dedup happens before any split.
    """
    before = len(df)
    subset = [c for c in TEXT_COLUMNS if c in df.columns]
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    print(f"deduplicate: removed {removed} exact duplicates, {len(df)} postings remain")
    return df


def train_test_frame(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split on the deduplicated frame."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[LABEL_COLUMN]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
