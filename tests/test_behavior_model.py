from datetime import datetime, timedelta, timezone
import math
import random

import pytest

from behavior_model import (
    ANOMALY_FEATURE_NAMES,
    BehaviorObservation,
    _features_for_observation,
    score_global_behavior,
    score_payer_behavior,
)


def _history(count: int = 60) -> list[BehaviorObservation]:
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


def test_features_include_log_elapsed_hours_since_previous_transaction():
    previous = BehaviorObservation(
        amount_cents=20_000,
        transaction_date=datetime(2026, 1, 1, 10, tzinfo=timezone.utc),
        device_id="device-known",
        latitude=-23.56,
        longitude=-46.65,
    )
    current = BehaviorObservation(
        amount_cents=20_000,
        transaction_date=previous.transaction_date + timedelta(hours=6),
        device_id="device-known",
        latitude=-23.56,
        longitude=-46.65,
    )

    features = _features_for_observation(current, [previous])

    assert len(features) == len(ANOMALY_FEATURE_NAMES)
    assert features[ANOMALY_FEATURE_NAMES.index("hours_since_last_transaction_log")] == pytest.approx(
        math.log1p(6)
    )
    assert features[ANOMALY_FEATURE_NAMES.index("has_previous_transaction")] == 1


def test_short_transaction_gaps_share_minute_precision():
    previous = BehaviorObservation(
        amount_cents=20_000,
        transaction_date=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        device_id="device-known",
        latitude=-23.56,
        longitude=-46.65,
    )
    current = BehaviorObservation(
        amount_cents=20_000,
        transaction_date=previous.transaction_date + timedelta(seconds=10),
        device_id="device-known",
        latitude=-23.56,
        longitude=-46.65,
    )

    features = _features_for_observation(current, [previous])

    assert features[ANOMALY_FEATURE_NAMES.index("hours_since_last_transaction_log")] == 0
    assert features[ANOMALY_FEATURE_NAMES.index("time_of_day_sin")] == pytest.approx(
        math.sin(2 * math.pi * 9 / 24)
    )
    assert features[ANOMALY_FEATURE_NAMES.index("time_of_day_cos")] == pytest.approx(
        math.cos(2 * math.pi * 9 / 24)
    )


def test_ten_second_gap_after_rapid_history_is_not_an_extreme_anomaly():
    start = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    history = [
        BehaviorObservation(
            amount_cents=20_000,
            transaction_date=start + timedelta(seconds=index * 5),
            device_id="device-known",
            latitude=-23.56,
            longitude=-46.65,
        )
        for index in range(50)
    ]
    current = BehaviorObservation(
        amount_cents=20_000,
        transaction_date=history[-1].transaction_date + timedelta(seconds=10),
        device_id="device-known",
        latitude=-23.56,
        longitude=-46.65,
    )

    score = score_payer_behavior(history, current)

    assert score is not None
    assert score < 0.30


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
        for index in range(220)
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
