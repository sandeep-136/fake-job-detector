"""Train, select and honestly evaluate models for fake job posting detection.

Usage:
    python src/train.py --csv data/fake_job_postings.csv

Evaluation protocol (deliberately strict):
  1. Duplicates are removed BEFORE any split (exact + case/punctuation
     variants). Reworded near-duplicates cannot be removed this way, so
     step 4 quantifies them instead of pretending they are gone.
  2. One stratified 80/20 split. The test set is then locked.
  3. Model selection uses ONLY the training set: 5-fold stratified
     cross-validation, models ranked by mean fraud-class F1.
  4. Leakage check: for every test posting, the maximum TF-IDF cosine
     similarity to any training posting is computed. Test postings with a
     near-twin in training (similarity >= 0.8) are flagged; metrics are
     reported on the full test set AND on the restricted subset without
     near-twins, so residual leakage is visible rather than hidden.
  5. The selected model is refit on the full training set and evaluated
     EXACTLY ONCE on the locked test set. Test numbers are never used to
     choose anything.

Metrics are reported for the fraud class (precision, recall, F1) plus
ROC-AUC - never accuracy alone: a model calling everything "real" scores
~95% accuracy here and detects nothing.

Outputs:
    models/model.joblib        selected pipeline, refit on all training data
    models/metrics.json        CV selection table + one-shot test results + leakage check
    docs/confusion_matrix.png  selected model, locked test set
    docs/roc_curve.png         selected model, locked test set
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from prepare import deduplicate, load_postings, train_test_frame, LABEL_COLUMN

NEAR_TWIN_THRESHOLD = 0.8


def make_models():
    tfidf = lambda: TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=100_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    return {
        # Majority-class dummy: the floor any real model must beat.
        "dummy_majority": Pipeline(
            [("tfidf", tfidf()), ("clf", DummyClassifier(strategy="most_frequent"))]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", tfidf()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
            ]
        ),
        "multinomial_nb": Pipeline(
            [("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]
        ),
        "linear_svc_calibrated": Pipeline(
            [
                ("tfidf", tfidf()),
                (
                    "clf",
                    CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto"), cv=3, n_jobs=-1),
                ),
            ]
        ),
    }


def select_by_cv(train_df, only=None):
    """Rank models by 5-fold stratified CV on the training set only.

    Supports --only MODEL for long sandboxed runs: each model's CV row is
    computed and cached separately (.cache/selection_parts.json); the final
    ranking is assembled once every model has a row.
    """
    parts_path = Path(".cache/selection_parts.json")
    parts = json.load(open(parts_path)) if parts_path.exists() else {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"f1": "f1", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
    models = make_models()
    # The dummy's CV row is analytic: it always predicts the majority class.
    if "dummy_majority" not in parts:
        parts["dummy_majority"] = {"model": "dummy_majority", "cv_f1_mean": 0.0, "cv_f1_std": 0.0,
                                   "cv_precision_mean": 0.0, "cv_precision_std": 0.0,
                                   "cv_recall_mean": 0.0, "cv_recall_std": 0.0,
                                   "cv_roc_auc_mean": 0.5, "cv_roc_auc_std": 0.0}
    todo = [only] if only else [n for n in models if n not in parts]
    for name in todo:
        if name in ("dummy_majority",) or name in parts and only is None:
            continue
        scores = cross_validate(
            models[name], train_df["text"], train_df[LABEL_COLUMN],
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        row = {"model": name}
        for key in scoring:
            vals = scores[f"test_{key}"]
            row[f"cv_{key}_mean"] = round(float(vals.mean()), 4)
            row[f"cv_{key}_std"] = round(float(vals.std()), 4)
        parts[name] = row
        print(f"CV {name}: f1={row['cv_f1_mean']}+/-{row['cv_f1_std']} "
              f"p={row['cv_precision_mean']} r={row['cv_recall_mean']} auc={row['cv_roc_auc_mean']}")
        json.dump(parts, open(parts_path, "w"), indent=2)
    table = [parts[n] for n in models if n in parts]
    json.dump(parts, open(parts_path, "w"), indent=2)
    if any(n not in parts for n in models):
        print("selection incomplete - remaining models still to run")
        return table, None
    candidates = [r for r in table if r["model"] != "dummy_majority"]
    best = max(candidates, key=lambda r: r["cv_f1_mean"])
    return table, best["model"]


def near_twin_analysis(train_texts, test_texts):
    """Max TF-IDF cosine similarity of each test posting to the training set.

    Reworded near-duplicates across the split would flatter the scores; this
    makes the overlap measurable. Uses a word-level TF-IDF space fit on
    training text only (separate from the model's own vectorizer).
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_train = normalize(vec.fit_transform(train_texts))
    X_test = normalize(vec.transform(test_texts))
    max_sim = np.asarray((X_test @ X_train.T).max(axis=1).todense()).ravel()
    return max_sim


def evaluate_once(pipe, test_df):
    y_true = test_df[LABEL_COLUMN].to_numpy()
    y_pred = pipe.predict(test_df["text"])
    y_score = pipe.predict_proba(test_df["text"])[:, 1]
    return {
        "precision_fraud": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_fraud": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_fraud": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_score)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(test_df)),
        "n_fraud": int(y_true.sum()),
    }, y_true, y_pred, y_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/fake_job_postings.csv")
    ap.add_argument("--out-models", default="models")
    ap.add_argument("--out-docs", default="docs")
    ap.add_argument("--stage", default="all", choices=["all", "split", "select", "final"],
                    help="run one stage (results cached in .cache/) or the whole pipeline")
    ap.add_argument("--only", default=None, choices=list(make_models()),
                    help="with --stage select: compute the CV row for one model only")
    args = ap.parse_args()

    Path(args.out_models).mkdir(exist_ok=True)
    Path(args.out_docs).mkdir(exist_ok=True)

    cache = Path(".cache")
    cache.mkdir(exist_ok=True)
    split_pkl = cache / "split.pkl"
    sel_json = cache / "selection.json"

    if args.stage in ("split", "all") or not split_pkl.exists():
        df = deduplicate(load_postings(args.csv))
        train_df, test_df = train_test_frame(df)
        joblib.dump({"train": train_df, "test": test_df, "rows": len(df),
                     "fraud_rate": float(df[LABEL_COLUMN].mean())}, split_pkl)
        print(f"split cached: train {len(train_df)}, test {len(test_df)}")
    cached = joblib.load(split_pkl)
    train_df, test_df = cached["train"], cached["test"]
    df_rows, fraud_rate = cached["rows"], cached["fraud_rate"]
    if args.stage == "split":
        return
    print(f"train: {len(train_df)} postings ({train_df[LABEL_COLUMN].mean():.2%} fraud)")
    print(f"test (locked): {len(test_df)} postings ({test_df[LABEL_COLUMN].mean():.2%} fraud)")

    if args.stage in ("select", "all") or not sel_json.exists():
        cv_table, best_name = select_by_cv(train_df, only=args.only)
        if best_name is None:
            return
        json.dump({"cv_selection_table": cv_table, "selected_model": best_name},
                  open(sel_json, "w"), indent=2)
        print(f"selection cached: {best_name}")
    sel = json.load(open(sel_json))
    cv_table, best_name = sel["cv_selection_table"], sel["selected_model"]
    if args.stage == "select":
        return
    print(f"selected by training-only CV: {best_name}")

    max_sim = near_twin_analysis(train_df["text"], test_df["text"])
    n_twins = int((max_sim >= NEAR_TWIN_THRESHOLD).sum())
    print(f"near-twin check: {n_twins}/{len(test_df)} test postings have a training "
          f"near-twin (cosine >= {NEAR_TWIN_THRESHOLD})")

    pipe = make_models()[best_name]
    pipe.fit(train_df["text"], train_df[LABEL_COLUMN])
    joblib.dump(pipe, Path(args.out_models) / "model.joblib")

    test_metrics, y_true, y_pred, y_score = evaluate_once(pipe, test_df)
    restricted_df = test_df[max_sim < NEAR_TWIN_THRESHOLD]
    restricted_metrics, _, _, _ = evaluate_once(pipe, restricted_df)

    report = {
        "protocol": [
            "duplicates removed before splitting (exact + case/punctuation variants)",
            "one stratified 80/20 split; test set locked",
            "model selection by 5-fold stratified CV on the training set only",
            "selected model refit on full training set, evaluated once on the locked test set",
            f"residual near-duplicate overlap quantified (max TF-IDF cosine >= {NEAR_TWIN_THRESHOLD}); restricted-subset metrics reported alongside",
        ],
        "dataset_rows_after_dedup": int(df_rows),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "fraud_rate": round(fraud_rate, 4),
        "cv_selection_table": cv_table,
        "selected_model": best_name,
        "final_test_metrics": test_metrics,
        "near_twin_analysis": {
            "threshold_cosine": NEAR_TWIN_THRESHOLD,
            "test_postings_with_train_near_twin": n_twins,
            "share_of_test": round(n_twins / len(test_df), 4),
            "restricted_test_metrics": restricted_metrics,
        },
    }
    (Path(args.out_models) / "metrics.json").write_text(json.dumps(report, indent=2))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["real", "fake"])
    disp.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"Confusion matrix - {best_name}\nlocked test set (selected via training-only CV)")
    plt.tight_layout()
    plt.savefig(Path(args.out_docs) / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - locked test set (selection via training-only CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.out_docs) / "roc_curve.png", dpi=150)
    plt.close()

    print(f"final test: {test_metrics}")
    print(f"restricted (no near-twins): {restricted_metrics}")


if __name__ == "__main__":
    main()
