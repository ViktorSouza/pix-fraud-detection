from datetime import datetime
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import ClassifierMixin

from errors.custom_errors import FraudModelUnavailable


MODEL_FORMAT_VERSION = 1
FEATURE_NAMES = (
    "amount_log_reais",
    "hour_sin",
    "hour_cos",
    "is_night",
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "pix_fraud_model.joblib"


def pix_features(amount_cents: int, transaction_date: datetime) -> np.ndarray:
    if transaction_date.tzinfo is None or transaction_date.utcoffset() is None:
        raise ValueError("transaction_date must include a timezone")

    hour = (
        transaction_date.hour
        + transaction_date.minute / 60
        + transaction_date.second / 3600
        + transaction_date.microsecond / 3_600_000_000
    )
    angle = 2 * math.pi * hour / 24
    return np.asarray(
        [
            [
                math.log1p(amount_cents / 100),
                math.sin(angle),
                math.cos(angle),
                float(hour >= 20 or hour < 6),
            ]
        ],
        dtype=np.float64,
    )


def paysim_features(steps: np.ndarray, amounts: np.ndarray) -> np.ndarray:
    hourly_steps = np.asarray(steps, dtype=np.float64)
    transaction_amounts = np.asarray(amounts, dtype=np.float64)
    hours = np.mod(hourly_steps, 24)
    angles = 2 * math.pi * hours / 24
    return np.column_stack(
        (
            np.log1p(np.maximum(transaction_amounts, 0)),
            np.sin(angles),
            np.cos(angles),
            ((hours >= 20) | (hours < 6)).astype(np.float64),
        )
    )


def decision_for_score(
    fraud_score: float,
    two_factor_threshold: float,
    reject_threshold: float,
) -> str:
    if not 0 <= fraud_score <= 1:
        raise ValueError("fraud_score must be between 0 and 1")
    if not 0 < two_factor_threshold < reject_threshold < 1:
        raise ValueError("decision thresholds must satisfy 0 < 2FA < rejection < 1")

    if fraud_score >= reject_threshold:
        return "rejected"
    if fraud_score >= two_factor_threshold:
        return "two_factor_required"
    return "approved"


def combine_decisions(
    fraud_score: float,
    rejection_reasons: list[str],
    two_factor_reasons: list[str],
    two_factor_threshold: float,
    reject_threshold: float,
) -> tuple[str, str, list[str]]:
    model_status = decision_for_score(
        fraud_score,
        two_factor_threshold=two_factor_threshold,
        reject_threshold=reject_threshold,
    )
    if model_status == "rejected":
        model_reasons = ["model_fraud_score_over_rejection_threshold"]
    elif model_status == "two_factor_required":
        model_reasons = ["model_fraud_score_requires_confirmation"]
    else:
        model_reasons = []

    if rejection_reasons or model_status == "rejected":
        status = "rejected"
    elif two_factor_reasons or model_status == "two_factor_required":
        status = "two_factor_required"
    else:
        status = "approved"

    if rejection_reasons:
        decision_layer = "level_2_hard_rule"
    elif two_factor_reasons and model_status == "approved":
        decision_layer = "level_4_rules"
    else:
        decision_layer = "paysim_prototype_model"
    return status, decision_layer, model_reasons


def score_transaction(
    amount_cents: int,
    transaction_date: datetime,
    model_path: Path,
) -> float:
    artifact = _load_artifact(str(model_path.resolve()))
    features = pix_features(amount_cents, transaction_date)
    model = artifact["model"]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Fraud model does not contain the fraud class")
    fraud_column = classes.index(1)
    fraud_score = float(model.predict_proba(features)[0, fraud_column])
    if not math.isfinite(fraud_score) or not 0 <= fraud_score <= 1:
        raise ValueError("Fraud model returned a score outside the 0-to-1 range")
    return fraud_score


@lru_cache(maxsize=4)
def _load_artifact(model_path: str) -> dict[str, Any]:
    try:
        artifact = joblib.load(model_path)
    except FileNotFoundError:
        raise FraudModelUnavailable(model_path) from None

    if not isinstance(artifact, dict):
        raise ValueError("Fraud model artifact must be a dictionary")
    if artifact.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported fraud model artifact version")
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Fraud model feature schema does not match this service")

    model = artifact.get("model")
    if (
        not isinstance(model, ClassifierMixin)
        or not hasattr(model, "predict_proba")
        or not hasattr(model, "classes_")
        or getattr(model, "n_features_in_", None) != len(FEATURE_NAMES)
    ):
        raise ValueError("Fraud model artifact has no compatible classifier")
    return artifact
