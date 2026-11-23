"""Evaluate fresh classifiers against the checked-in synthetic telemetry."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "telemetry.csv"
OUTPUT_DIR = ROOT_DIR / "evaluation"
FEATURES = ["cpu", "memory", "disk", "temperature", "network"]
SEED = 42
TEST_SIZE = 0.2
FOLDS = 5
THRESHOLD = 0.5
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 100,
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.3,
    "random_state": SEED,
    "n_jobs": 1,
}
EVALUATION_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib")


def class_counts(labels):
    positives = int(labels.sum())
    return {
        "rows": len(labels),
        "positives": positives,
        "negatives": len(labels) - positives,
        "positive_prevalence": positives / len(labels),
    }


def predict_scores(estimators, x_train, y_train, x_test):
    return {
        name: clone(estimator).fit(x_train, y_train).predict_proba(x_test)[:, 1]
        for name, estimator in estimators.items()
    }


def measure(labels, scores):
    predicted = (scores >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def cross_validate(estimators, features, labels):
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for index, (fit_rows, validation_rows) in enumerate(splitter.split(features, labels), start=1):
        validation_labels = labels.iloc[validation_rows]
        scores = predict_scores(
            estimators, features.iloc[fit_rows], labels.iloc[fit_rows], features.iloc[validation_rows]
        )
        folds.append({
            "fold": index,
            "training": class_counts(labels.iloc[fit_rows]),
            "validation": class_counts(validation_labels),
            "metrics": {name: measure(validation_labels, probabilities) for name, probabilities in scores.items()},
        })

    summary = {}
    for name in estimators:
        summary[name] = {}
        for metric in folds[0]["metrics"][name]:
            if metric == "confusion_matrix":
                continue
            values = [fold["metrics"][name][metric] for fold in folds]
            summary[name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": min(values),
                "max": max(values),
            }
    return {"folds": folds, "summary": summary}


def plot_curves(labels, scores, metrics):
    figure = Figure(figsize=(10, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    roc, pr = figure.subplots(1, 2)
    for name, title, style in (("xgboost", "XGBoost", "-"), ("majority", "Majority baseline", "--")):
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores[name])
        precision, recall, _ = precision_recall_curve(labels, scores[name])
        roc.plot(
            false_positive_rate, true_positive_rate, style,
            label=f"{title} (AUC {metrics[name]['roc_auc']:.4f})",
        )
        pr.step(
            recall, precision, where="post", linestyle=style,
            label=f"{title} (AP {metrics[name]['average_precision']:.4f})",
        )

    roc.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    pr.set(title="Precision–recall curve", xlabel="Recall", ylabel="Precision")
    for axis in (roc, pr):
        axis.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.05))
        axis.grid(alpha=0.25)
    roc.legend(loc="lower right")
    pr.legend(loc="center right")
    figure.suptitle(f"Synthetic telemetry holdout · {len(labels):,} rows · {int(labels.sum())} failures")
    figure.savefig(OUTPUT_DIR / "holdout_curves.png", dpi=160)


def plot_confusion(metrics):
    figure = Figure(figsize=(8, 4.2), layout="constrained")
    FigureCanvasAgg(figure)
    classes = ["No failure", "Failure"]
    for axis, name, title in zip(figure.subplots(1, 2), ("xgboost", "majority"), ("XGBoost", "Majority baseline")):
        counts = np.asarray(metrics[name]["confusion_matrix"])
        fractions = counts / counts.sum(axis=1, keepdims=True)
        axis.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for (row, col), count in np.ndenumerate(counts):
            fraction = fractions[row, col]
            color = "white" if fraction > 0.55 else "#10243b"
            axis.text(col, row, f"{count:,}\n({fraction:.1%})", ha="center", va="center", color=color)
        axis.set(
            xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
            title=title, xlabel="Predicted label", ylabel="Actual label",
        )
    figure.suptitle(f"Holdout confusion matrices at threshold {THRESHOLD:.2f}\nPercentages are within each actual class")
    figure.savefig(OUTPUT_DIR / "holdout_confusion.png", dpi=160)


def main():
    data = pd.read_csv(DATA_PATH)
    features, labels = data[FEATURES], data["failure"]
    if features.duplicated().any():
        raise ValueError("Duplicate feature rows require a grouped split before evaluation.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )
    estimators = {
        "xgboost": XGBClassifier(**XGB_PARAMS),
        "majority": DummyClassifier(strategy="most_frequent"),
    }
    cv_results = cross_validate(estimators, x_train, y_train)
    holdout_scores = predict_scores(estimators, x_train, y_train, x_test)
    holdout_metrics = {name: measure(y_test, scores) for name, scores in holdout_scores.items()}

    results = {
        "dataset": {
            "path": DATA_PATH.relative_to(ROOT_DIR).as_posix(),
            "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "features": FEATURES,
            **class_counts(labels),
        },
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in EVALUATION_PACKAGES},
        },
        "configuration": {
            "seed": SEED,
            "holdout_fraction": TEST_SIZE,
            "stratified_cv_folds_on_training_partition": FOLDS,
            "positive_label": 1,
            "classification_threshold": THRESHOLD,
            "confusion_matrix_label_order": [0, 1],
            "xgboost": XGB_PARAMS,
            "majority": {"strategy": "most_frequent"},
        },
        "split": {"training": class_counts(y_train), "holdout": class_counts(y_test)},
        "holdout_metrics": holdout_metrics,
        "cross_validation": cv_results,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    predictions = pd.DataFrame({
        "failure": y_test,
        **{f"{name}_score": scores for name, scores in holdout_scores.items()},
    })
    predictions.sort_index().to_csv(
        OUTPUT_DIR / "holdout_predictions.csv", index_label="dataset_row", float_format="%.17g", lineterminator="\n"
    )
    plot_curves(y_test, holdout_scores, holdout_metrics)
    plot_confusion(holdout_metrics)

    print(json.dumps(holdout_metrics, indent=2))
    print(f"Saved metrics, holdout predictions, and plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
