"""Train and honestly evaluate baseline models for fake job posting detection.

Usage:
    python src/train.py --csv data/fake_job_postings.csv

Outputs:
    models/model.joblib        - best pipeline (TF-IDF + classifier), for the M2 API
    models/metrics.json        - per-model metrics on the held-out test set
    docs/confusion_matrix.png  - confusion matrix of the selected model
    docs/roc_curve.png         - ROC curves of all compared models

Evaluation rules this script follows:
  * exact-duplicate postings are removed BEFORE the train/test split
    (duplicates straddling the split inflate scores);
  * the split is stratified - the dataset is ~5% fraudulent;
  * metrics are reported for the fraud class (precision, recall, F1) plus
    ROC-AUC, never accuracy alone: a model that calls everything "real"
    scores ~95% accuracy here and detects nothing.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from prepare import deduplicate, load_postings, train_test_frame, LABEL_COLUMN


def make_models():
    tfidf = lambda: TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
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
        # Calibrated so it can output probabilities like the others.
        "linear_svc_calibrated": Pipeline(
            [
                ("tfidf", tfidf()),
                (
                    "clf",
                    CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto"), cv=3),
                ),
            ]
        ),
    }


def evaluate(name, pipe, test_df):
    y_true = test_df[LABEL_COLUMN].to_numpy()
    y_pred = pipe.predict(test_df["text"])
    y_score = pipe.predict_proba(test_df["text"])[:, 1]
    return {
        "model": name,
        "precision_fraud": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_fraud": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_fraud": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_score)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }, y_true, y_pred, y_score


def top_terms(pipe, n=25):
    """Highest-weight terms of the logistic model (inspection, not explanation)."""
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    vocab = vectorizer.get_feature_names_out()
    weights = clf.coef_[0]
    top = weights.argsort()[::-1][:n]
    return [{"term": str(vocab[i]), "weight": round(float(weights[i]), 3)} for i in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/fake_job_postings.csv")
    ap.add_argument("--out-models", default="models")
    ap.add_argument("--out-docs", default="docs")
    args = ap.parse_args()

    Path(args.out_models).mkdir(exist_ok=True)
    Path(args.out_docs).mkdir(exist_ok=True)

    df = deduplicate(load_postings(args.csv))
    train_df, test_df = train_test_frame(df)
    print(f"train: {len(train_df)} postings ({train_df[LABEL_COLUMN].mean():.2%} fraud)")
    print(f"test:  {len(test_df)} postings ({test_df[LABEL_COLUMN].mean():.2%} fraud)")

    results, curves = [], {}
    fitted = {}
    for name, pipe in make_models().items():
        pipe.fit(train_df["text"], train_df[LABEL_COLUMN])
        metrics, y_true, y_pred, y_score = evaluate(name, pipe, test_df)
        results.append(metrics)
        curves[name] = (y_true, y_score)
        fitted[name] = (pipe, metrics, y_true, y_pred)
        print(f"{name}: {metrics}")

    # Select the model with the best fraud-class F1 among non-dummy models.
    candidates = [r for r in results if r["model"] != "dummy_majority"]
    best = max(candidates, key=lambda r: r["f1_fraud"])
    best_name = best["model"]
    pipe, metrics, y_true, y_pred = fitted[best_name]

    joblib.dump(pipe, Path(args.out_models) / "model.joblib")

    report = {
        "dataset_rows_after_dedup": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "fraud_rate": round(float(df[LABEL_COLUMN].mean()), 4),
        "selected_model": best_name,
        "selection_metric": "f1_fraud (fraud class, held-out test set)",
        "results": results,
        "top_terms_selected_model": top_terms(pipe) if best_name == "logistic_regression" else [],
    }
    (Path(args.out_models) / "metrics.json").write_text(json.dumps(report, indent=2))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["real", "fake"])
    disp.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"Confusion matrix - {best_name} (held-out test set)")
    plt.tight_layout()
    plt.savefig(Path(args.out_docs) / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    for name, (yt, ys) in curves.items():
        if name == "dummy_majority":
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (held-out test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.out_docs) / "roc_curve.png", dpi=150)
    plt.close()

    print(f"selected: {best_name}")


if __name__ == "__main__":
    main()
