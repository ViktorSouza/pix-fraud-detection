from datetime import datetime, timedelta, timezone
import math
import random

import pytest

from behavior_model import (
    BehaviorObservation,
    DIRECTIONAL_RISK_NAMES,
    _features_for_observation,
    evaluate_global_behavior,
    evaluate_payer_behavior,
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

    assert features["hours_since_last_transaction_log"] == pytest.approx(math.log1p(6))


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

    assert features["hours_since_last_transaction_log"] == 0


def test_ten_second_gap_after_rapid_history_does_not_raise_risk():
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

    result = evaluate_payer_behavior(history, current)

    assert result is not None
    assert result.score < 0.30
    assert result.signals["shorter_transaction_interval"] == 0


def test_faster_interval_increases_risk_but_slower_interval_is_neutral():
    start = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    history = [
        BehaviorObservation(
            amount_cents=20_000,
            transaction_date=start + timedelta(hours=index),
            device_id="device-known",
            latitude=-23.56,
            longitude=-46.65,
        )
        for index in range(60)
    ]
    faster = BehaviorObservation(
        amount_cents=20_000,
        transaction_date=history[-1].transaction_date + timedelta(minutes=5),
        device_id="device-known",
        latitude=-23.56,
        longitude=-46.65,
    )
    slower = faster._replace(
        transaction_date=history[-1].transaction_date + timedelta(hours=3)
    )

    fast_result = evaluate_payer_behavior(history, faster)
    slow_result = evaluate_payer_behavior(history, slower)

    assert fast_result is not None and slow_result is not None
    assert fast_result.signals["shorter_transaction_interval"] > 0.30
    assert slow_result.signals["shorter_transaction_interval"] == 0


def test_higher_amount_increases_risk_but_lower_amount_is_neutral():
    history = _history()
    current = BehaviorObservation(
        amount_cents=100_000,
        transaction_date=history[-1].transaction_date + timedelta(days=1),
        device_id="device-known",
        latitude=history[-1].latitude,
        longitude=history[-1].longitude,
    )
    lower = current._replace(amount_cents=1_000)

    higher_result = evaluate_payer_behavior(history, current)
    lower_result = evaluate_payer_behavior(history, lower)

    assert higher_result is not None and lower_result is not None
    assert higher_result.signals["amount_increase"] > 0.30
    assert lower_result.signals["amount_increase"] == 0
    assert "amount_increase" in DIRECTIONAL_RISK_NAMES


def test_new_device_is_a_positive_risk_signal():
    history = _history()
    current = BehaviorObservation(
        amount_cents=20_000,
        transaction_date=history[-1].transaction_date + timedelta(days=1),
        device_id="new-device",
        latitude=history[-1].latitude,
        longitude=history[-1].longitude,
    )

    result = evaluate_payer_behavior(history, current)

    assert result is not None
    assert result.signals["new_device"] == 0.4


def test_higher_travel_speed_increases_risk_only_in_the_suspicious_direction():
    history = _history()
    previous = history[-1]
    faster = BehaviorObservation(
        amount_cents=previous.amount_cents,
        transaction_date=previous.transaction_date + timedelta(hours=1),
        device_id=previous.device_id,
        latitude=40.71,
        longitude=-74.0,
    )
    slower = BehaviorObservation(
        amount_cents=previous.amount_cents,
        transaction_date=previous.transaction_date + timedelta(hours=1),
        device_id=previous.device_id,
        latitude=previous.latitude,
        longitude=previous.longitude,
    )

    fast_result = evaluate_payer_behavior(history, faster)
    slow_result = evaluate_payer_behavior(history, slower)

    assert fast_result is not None and slow_result is not None
    assert fast_result.signals["higher_travel_speed"] > 0.30
    assert slow_result.signals["higher_travel_speed"] == 0


def test_global_model_uses_directional_signals_and_current_payer_history():
    start = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    global_history = []
    payer_history = []
    for index in range(220):
        payer = f"payer-{index % 5}"
        observation = BehaviorObservation(
            amount_cents=20_000,
            transaction_date=start + timedelta(hours=index),
            device_id=f"device-{index % 5}",
            latitude=-23.56,
            longitude=-46.65,
        )
        global_history.append((payer, observation))
        if payer == "payer-0":
            payer_history.append(observation)

    current = BehaviorObservation(
        amount_cents=20_000,
        transaction_date=payer_history[-1].transaction_date + timedelta(minutes=1),
        device_id=payer_history[-1].device_id,
        latitude=-23.56,
        longitude=-46.65,
    )
    result = evaluate_global_behavior(
        global_history,
        current_payer="payer-0",
        current=current,
        current_payer_history=payer_history,
    )

    assert result is not None
    assert result.signals["shorter_transaction_interval"] > 0.3
    assert result.signals["amount_increase"] == 0
    assert 0 <= result.score <= 1


def test_low_amount_and_late_hour_do_not_raise_risk_by_themselves():
    history = _history()
    current = BehaviorObservation(
        amount_cents=2_000,
        transaction_date=history[-1].transaction_date + timedelta(days=1, hours=12),
        device_id="device-known",
        latitude=history[-1].latitude,
        longitude=history[-1].longitude,
    )

    result = evaluate_payer_behavior(history, current)

    assert result is not None
    assert result.signals["amount_increase"] == 0
    assert result.signals["shorter_transaction_interval"] == 0
    assert result.score < 0.30


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
