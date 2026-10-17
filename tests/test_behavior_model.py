from datetime import datetime, timedelta, timezone
import random

from behavior_model import (
    BehaviorObservation,
    score_global_behavior,
    score_payer_behavior,
)


def _history(count: int = 30) -> list[BehaviorObservation]:
    randomizer = random.Random(7)
    start = datetime(2026, 1, 1, 14, tzinfo=timezone.utc)
    return [
        BehaviorObservation(
            amount_cents=randomizer.randrange(18_000, 23_000),
            transaction_date=start
            + timedelta(days=index, hours=randomizer.randrange(-2, 3)),
            device_id="device-known",
            latitude=-23.56 + randomizer.uniform(-0.01, 0.01),
            longitude=-46.65 + randomizer.uniform(-0.01, 0.01),
        )
        for index in range(count)
    ]


def test_payer_model_scores_unusual_low_value_and_location_as_anomalous():
    history = _history()
    current = BehaviorObservation(
        amount_cents=2_000,
        transaction_date=history[-1].transaction_date + timedelta(days=1, hours=12),
        device_id="device-new",
        latitude=40.71,
        longitude=-74.0,
    )

    score = score_payer_behavior(history, current)

    assert score is not None
    assert score >= 0.70


def test_payer_model_keeps_a_typical_transaction_score_low():
    history = _history()
    current = BehaviorObservation(
        amount_cents=20_500,
        transaction_date=history[-1].transaction_date + timedelta(days=1, hours=1),
        device_id="device-known",
        latitude=-23.56,
        longitude=-46.65,
    )

    score = score_payer_behavior(history, current)

    assert score is not None
    assert score < 0.30


def test_payer_model_defers_to_global_fallback_with_insufficient_history():
    score = score_payer_behavior(
        _history(count=10),
        BehaviorObservation(
            amount_cents=20_000,
            transaction_date=datetime(2026, 2, 1, 14, tzinfo=timezone.utc),
            device_id="device-known",
            latitude=-23.56,
            longitude=-46.65,
        ),
    )

    assert score is None


def test_global_behavior_model_can_score_a_payer_without_history():
    start = datetime(2026, 1, 1, 14, tzinfo=timezone.utc)
    global_history = [
        (
            f"payer-{index % 5}",
            BehaviorObservation(
                amount_cents=20_000,
                transaction_date=start + timedelta(days=index),
                device_id=f"device-{index % 5}",
                latitude=-23.56 + (index % 5) * 0.001,
                longitude=-46.65,
            ),
        )
        for index in range(30)
    ]

    score = score_global_behavior(
        global_history,
        current_payer="new-payer",
        current=BehaviorObservation(
            amount_cents=2_000,
            transaction_date=start + timedelta(days=31, hours=12),
            device_id="new-device",
            latitude=40.71,
            longitude=-74.0,
        ),
    )

    assert score is not None
    assert 0 <= score <= 1
