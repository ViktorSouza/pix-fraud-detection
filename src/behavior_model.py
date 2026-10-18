from datetime import datetime
import math
from typing import NamedTuple

import numpy as np
from sklearn.ensemble import IsolationForest

from utils.geolocation import calculate_location_speed_kmh

MIN_PAYER_HISTORY = 50
MIN_GLOBAL_HISTORY = 200
MAX_PAYER_HISTORY = 300
ANOMALY_FEATURE_NAMES = (
    "amount_log_reais",
    "time_of_day_sin",
    "time_of_day_cos",
    "hours_since_last_transaction_log",
    "has_previous_transaction",
    "travel_speed_log_kmh",
    "has_location_velocity",
    "is_new_device",
)


class BehaviorObservation(NamedTuple):
    amount_cents: int
    transaction_date: datetime
    device_id: str
    latitude: float | None
    longitude: float | None


def _features_for_observation(
    observation: BehaviorObservation,
    previous_observations: list[BehaviorObservation],
) -> np.ndarray:
    hour = (
        observation.transaction_date.hour
        + observation.transaction_date.minute / 60
        + observation.transaction_date.second / 3600
        + observation.transaction_date.microsecond / 3_600_000_000
    )
    time_of_day_block_hour = math.floor(hour / 6) * 6 + 3
    angle = 2 * math.pi * time_of_day_block_hour / 24
    previous_observation = previous_observations[-1] if previous_observations else None
    hours_since_last_transaction = (
        math.floor(
            max(
                0.0,
                (
                    observation.transaction_date
                    - previous_observation.transaction_date
                ).total_seconds(),
            )
            / 60,
        )
        / 60
        if previous_observation is not None
        else 0.0
    )
    location_speed = None
    if observation.latitude is not None and observation.longitude is not None:
        previous_location = next(
            (
                item
                for item in reversed(previous_observations)
                if item.latitude is not None and item.longitude is not None
            ),
            None,
        )
        if previous_location is not None:
            movement = calculate_location_speed_kmh(
                previous_latitude=previous_location.latitude,
                previous_longitude=previous_location.longitude,
                previous_date=previous_location.transaction_date,
                current_latitude=observation.latitude,
                current_longitude=observation.longitude,
                current_date=observation.transaction_date,
            )
            if movement is not None:
                _, location_speed = movement

    device_seen = any(
        item.device_id == observation.device_id for item in previous_observations
    )
    return np.asarray(
        [
            math.log1p(observation.amount_cents / 100),
            math.sin(angle),
            math.cos(angle),
            math.log1p(hours_since_last_transaction),
            float(previous_observation is not None),
            math.log1p(location_speed or 0),
            float(location_speed is not None),
            float(bool(previous_observations) and not device_seen),
        ],
        dtype=np.float64,
    )


def _history_matrix(
    history: list[BehaviorObservation],
) -> tuple[np.ndarray, list[BehaviorObservation]]:
    ordered_history = sorted(history, key=lambda item: item.transaction_date)[
        -MAX_PAYER_HISTORY:
    ]
    rows = [
        _features_for_observation(observation, ordered_history[:index])
        for index, observation in enumerate(ordered_history)
    ]
    return np.asarray(rows, dtype=np.float64), ordered_history


def _score_isolation_forest(
    training_features: np.ndarray, current_features: np.ndarray
) -> float:
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )
    model.fit(training_features)

    training_normality = model.decision_function(training_features)
    current_normality = float(
        model.decision_function(current_features.reshape(1, -1))[0]
    )
    median_normality = float(np.median(training_normality))
    mad = float(np.median(np.abs(training_normality - median_normality)))
    spread = max(1.4826 * mad, float(np.std(training_normality)) * 0.25, 0.02)
    anomaly_z_score = (median_normality - current_normality) / spread
    anomaly_z_score = float(np.clip(anomaly_z_score, -20, 20))
    return float(1 / (1 + math.exp(-2 * (anomaly_z_score - 2))))


def score_payer_behavior(
    history: list[BehaviorObservation],
    current: BehaviorObservation,
) -> float | None:
    if len(history) < MIN_PAYER_HISTORY:
        return None

    training_features, ordered_history = _history_matrix(history)
    current_features = _features_for_observation(current, ordered_history)
    return _score_isolation_forest(training_features, current_features)


def score_global_behavior(
    history: list[tuple[str, BehaviorObservation]],
    current_payer: str,
    current: BehaviorObservation,
) -> float | None:
    if len(history) < MIN_GLOBAL_HISTORY:
        return None

    payer_histories: dict[str, list[BehaviorObservation]] = {}
    training_rows = []
    for payer, observation in history:
        previous = payer_histories.setdefault(payer, [])
        training_rows.append(_features_for_observation(observation, previous))
        previous.append(observation)

    current_features = _features_for_observation(
        current, payer_histories.get(current_payer, [])
    )
    return _score_isolation_forest(
        np.asarray(training_rows, dtype=np.float64), current_features
    )
