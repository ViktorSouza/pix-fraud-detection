import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from fraud_model import LEGACY_FEATURE_NAMES, LEGACY_MODEL_FORMAT_VERSION, paysim_features


def train_model(
    dataset_path: Path,
    output_path: Path,
    max_rows: int,
    random_state: int,
) -> dict[str, float | int]:
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2")

    data = pd.read_csv(
        dataset_path,
        usecols=["step", "amount", "isFraud"],
        dtype={"step": "float32", "amount": "float32", "isFraud": "int8"},
    )
    if data.empty:
        raise ValueError("The training CSV contains no transactions")
    if data["isFraud"].isna().any() or not set(data["isFraud"].unique()).issubset({0, 1}):
        raise ValueError("PaySim isFraud must contain only 0 and 1 labels")
    if data["step"].isna().any() or data["amount"].isna().any():
        raise ValueError("PaySim step and amount must not contain missing values")
    if not np.isfinite(data[["step", "amount"]].to_numpy()).all():
        raise ValueError("PaySim step and amount must be finite numbers")
    if (data["step"] < 0).any() or (data["amount"] < 0).any():
        raise ValueError("PaySim step and amount must not be negative")

    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=random_state)

    labels = data["isFraud"].to_numpy(dtype=np.int8)
    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < 10:
        raise ValueError("Training requires at least ten rows from each fraud class")

    features = paysim_features(
        data["step"].to_numpy(),
        data["amount"].to_numpy(),
    )
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
    metrics = {
        "roc_auc": float(roc_auc_score(test_labels, fraud_scores)),
        "average_precision": float(average_precision_score(test_labels, fraud_scores)),
        "brier_score": float(brier_score_loss(test_labels, fraud_scores)),
        "test_rows": int(len(test_labels)),
        "test_fraud_rate": float(test_labels.mean()),
    }
    artifact = {
        "format_version": LEGACY_MODEL_FORMAT_VERSION,
        "feature_names": LEGACY_FEATURE_NAMES,
        "dataset": "PaySim1",
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
        description="Train the academic Pix fraud-score prototype with PaySim1."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the PaySim CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pix_fraud_model.joblib"),
        help="Path where the model artifact will be written",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1_000_000,
        help="Maximum number of randomly sampled rows to train on",
    )
    parser.add_argument("--random-state", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.max_rows < 2:
        parser.error("--max-rows must be at least 2")

    metrics = train_model(
        dataset_path=arguments.dataset,
        output_path=arguments.output,
        max_rows=arguments.max_rows,
        random_state=arguments.random_state,
    )
    print(json.dumps({"model": str(arguments.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
