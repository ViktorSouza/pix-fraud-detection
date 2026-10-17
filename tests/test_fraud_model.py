from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fraud_model import (
    combine_decisions,
    decision_for_score,
    paysim_features,
    pix_features,
    score_transaction,
)
from errors.custom_errors import FraudModelUnavailable
from train_paysim_model import train_model


def test_pix_and_paysim_features_share_the_same_schema():
    pix = pix_features(
        amount_cents=25_000,
        transaction_date=datetime(2026, 10, 17, 21, tzinfo=timezone.utc),
    )
    paysim = paysim_features(
        steps=np.asarray([21]),
        amounts=np.asarray([250]),
    )

    np.testing.assert_allclose(pix, paysim)


def test_pix_features_reject_naive_timestamps():
    with pytest.raises(ValueError, match="must include a timezone"):
        pix_features(25_000, datetime(2026, 10, 17, 21))


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "approved"),
        (0.299999, "approved"),
        (0.30, "two_factor_required"),
        (0.699999, "two_factor_required"),
        (0.70, "rejected"),
        (1.0, "rejected"),
    ],
)
def test_decision_thresholds(score, expected):
    assert decision_for_score(score, 0.30, 0.70) == expected


def test_existing_rules_can_only_escalate_model_decisions():
    status, layer, model_reasons = combine_decisions(
        fraud_score=0.1,
        rejection_reasons=[],
        two_factor_reasons=["new_device_for_payer"],
        two_factor_threshold=0.30,
        reject_threshold=0.70,
    )

    assert status == "two_factor_required"
    assert layer == "level_4_rules"
    assert model_reasons == []


def test_model_rejection_can_escalate_a_transaction():
    status, layer, model_reasons = combine_decisions(
        fraud_score=0.8,
        rejection_reasons=[],
        two_factor_reasons=[],
        two_factor_threshold=0.30,
        reject_threshold=0.70,
    )

    assert status == "rejected"
    assert layer == "paysim_prototype_model"
    assert model_reasons == ["model_fraud_score_over_rejection_threshold"]


def test_hard_rejection_remains_authoritative():
    status, layer, model_reasons = combine_decisions(
        fraud_score=0.1,
        rejection_reasons=["amount_over_transaction_limit"],
        two_factor_reasons=[],
        two_factor_threshold=0.30,
        reject_threshold=0.70,
    )

    assert status == "rejected"
    assert layer == "level_2_hard_rule"
    assert model_reasons == []


def test_missing_artifact_is_an_explicit_service_error(tmp_path):
    with pytest.raises(FraudModelUnavailable):
        score_transaction(
            25_000,
            datetime(2026, 10, 17, 21, tzinfo=timezone.utc),
            tmp_path / "missing.joblib",
        )


def test_training_creates_a_loadable_model(tmp_path):
    dataset_path = tmp_path / "paysim_sample.csv"
    output_path = tmp_path / "model.joblib"
    pd.DataFrame(
        [
            {
                "step": step,
                "amount": float((step * 37) % 500 + 1),
                "isFraud": int(step % 7 == 0),
            }
            for step in range(1, 281)
        ]
    ).to_csv(dataset_path, index=False)

    metrics = train_model(dataset_path, output_path, max_rows=1_000, random_state=42)

    assert output_path.is_file()
    assert metrics["test_rows"] == 56
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1
    score = score_transaction(
        25_000,
        datetime(2026, 10, 17, 21, tzinfo=timezone.utc),
        output_path,
    )
    assert 0 <= score <= 1
