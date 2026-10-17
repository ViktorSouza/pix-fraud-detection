import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from fraud_model import FEATURE_NAMES, MODEL_FORMAT_VERSION


LABEL_COLUMN = "is_fraud"
BINARY_FEATURE_NAMES = (
    "is_night",
    "has_prior_approved_history",
    "is_new_destination",
    "is_new_device",
    "has_location_velocity",
)


def train_model(
    dataset_path: Path,
    output_path: Path,
    random_state: int = 42,
    two_factor_threshold: float = 0.30,
) -> dict[str, float | int | None]:
    if not 0 < two_factor_threshold < 1:
        raise ValueError("two_factor_threshold must be between 0 and 1")

    data = pd.read_csv(dataset_path, usecols=[*FEATURE_NAMES, LABEL_COLUMN])
    if data.empty:
        raise ValueError("The labeled Pix CSV contains no transactions")

    labels = data[LABEL_COLUMN].to_numpy()
    if not np.isfinite(labels).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1 labels")
    labels = labels.astype(np.int8)
    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < 10:
        raise ValueError("Training requires at least ten rows from each fraud class")

    features_frame = data.loc[:, FEATURE_NAMES].astype(np.float64)
    features = features_frame.to_numpy()
    if not np.isfinite(features).all():
        raise ValueError("Pix model features must not contain missing or non-finite values")
    non_negative_features = (
        "prior_approved_count_24h",
        "prior_approved_amount_log_reais",
        "prior_approved_count_10m",
        "location_speed_kmh",
    )
    if (features_frame.loc[:, non_negative_features] < 0).any().any():
        raise ValueError("History and location-speed features must not be negative")
    for name in BINARY_FEATURE_NAMES:
        if not set(features_frame[name].unique()).issubset({0, 1}):
            raise ValueError(f"{name} must contain only 0 and 1")

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=random_state,
    )
    model.fit(train_features, train_labels)
    fraud_scores = model.predict_proba(test_features)[:, list(model.classes_).index(1)]
    low_value_mask = test_features[:, FEATURE_NAMES.index("amount_log_reais")] <= np.log1p(200)
    low_value_fraud_mask = (test_labels == 1) & low_value_mask
    low_value_fraud_count = int(low_value_fraud_mask.sum())
    low_value_recall = (
        float((fraud_scores[low_value_fraud_mask] >= two_factor_threshold).mean())
        if low_value_fraud_count
        else None
    )
    metrics = {
        "roc_auc": float(roc_auc_score(test_labels, fraud_scores)),
        "average_precision": float(average_precision_score(test_labels, fraud_scores)),
        "brier_score": float(brier_score_loss(test_labels, fraud_scores)),
        "test_rows": int(len(test_labels)),
        "test_fraud_rate": float(test_labels.mean()),
        "low_value_fraud_rows": low_value_fraud_count,
        "low_value_recall_at_two_factor_threshold": low_value_recall,
    }
    artifact = {
        "format_version": MODEL_FORMAT_VERSION,
        "feature_names": FEATURE_NAMES,
        "dataset": "labeled_pix_transactions",
        "training_rows": int(len(data)),
        "training_fraud_rate": float(labels.mean()),
        "metrics": metrics,
        "model": model,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Pix fraud model from labeled, server-derived features."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the labeled Pix CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pix_fraud_model.joblib"),
        help="Path where the model artifact will be written",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--two-factor-threshold", type=float, default=0.30)
    arguments = parser.parse_args()

    metrics = train_model(
        dataset_path=arguments.dataset,
        output_path=arguments.output,
        random_state=arguments.random_state,
        two_factor_threshold=arguments.two_factor_threshold,
    )
    print(json.dumps({"model": str(arguments.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
