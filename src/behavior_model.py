from datetime import datetime
import math
from typing import NamedTuple, TypedDict

import numpy as np

from utils.geolocation import calculate_location_speed_kmh

MIN_PAYER_HISTORY = 50
MIN_GLOBAL_HISTORY = 200
MAX_PAYER_HISTORY = 30
DIRECTIONAL_RISK_NAMES = (
    "amount_increase",
    "shorter_transaction_interval",
    "higher_travel_speed",
    "new_device",
)
_NEW_DEVICE_RISK = 0.4


class _DirectionalFeatures(TypedDict):
    amount_log_reais: float
    hours_since_last_transaction_log: float | None
    travel_speed_log_kmh: float | None


class BehaviorObservation(NamedTuple):
    amount_cents: int
    transaction_date: datetime
    device_id: str
    latitude: float | None
    longitude: float | None


class DirectionalRiskScore(NamedTuple):
    score: float
    signals: dict[str, float]


def _features_for_observation(
    observation: BehaviorObservation,
    previous_observations: list[BehaviorObservation],
) -> _DirectionalFeatures:
    previous_observation = previous_observations[-1] if previous_observations else None
    hours_since_last_transaction = None
    if previous_observation is not None:
        elapsed_minutes = math.floor(
            max(
                0.0,
                (
                    observation.transaction_date - previous_observation.transaction_date
                ).total_seconds(),
            )
            / 60
        )
        hours_since_last_transaction = math.log1p(elapsed_minutes / 60)

    location_speed = None
    if observation.latitude is not None and observation.longitude is not None:
        for previous_location in reversed(previous_observations):
            if (
                previous_location.latitude is None
                or previous_location.longitude is None
            ):
                continue
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
            break

    return {
        "amount_log_reais": math.log1p(observation.amount_cents / 100),
        "hours_since_last_transaction_log": hours_since_last_transaction,
        "travel_speed_log_kmh": (
            math.log1p(location_speed) if location_speed is not None else None
        ),
    }


def _upper_directional_risk(reference: list[float], current: float) -> float:
    center = float(np.median(reference))
    if current <= center:
        return 0.0

    deviations = np.abs(np.asarray(reference, dtype=np.float64) - center)
    spread = max(
        1.4826 * float(np.median(deviations)),
        float(np.std(reference)) * 0.25,
        0.02,
    )
    directional_z_score = (current - center) / spread
    return float(1 / (1 + math.exp(-2 * (directional_z_score - 2))))


def _lower_directional_risk(reference: list[float], current: float) -> float:
    center = float(np.median(reference))
    if current >= center:
        return 0.0

    deviations = np.abs(np.asarray(reference, dtype=np.float64) - center)
    spread = max(
        1.4826 * float(np.median(deviations)),
        float(np.std(reference)) * 0.25,
        0.02,
    )
    directional_z_score = (center - current) / spread
    return float(1 / (1 + math.exp(-2 * (directional_z_score - 2))))


def _combine_directional_risks(signals: dict[str, float]) -> float:
    return float(1 - math.prod(1 - score for score in signals.values()))


def _score_directional_behavior(
    history_features: list[_DirectionalFeatures],
    current_features: _DirectionalFeatures,
    is_new_device: bool,
) -> DirectionalRiskScore:
    amount_history = [features["amount_log_reais"] for features in history_features]
    interval_history = [
        value
        for features in history_features
        if (value := features["hours_since_last_transaction_log"]) is not None
    ]
    speed_history = [
        value
        for features in history_features
        if (value := features["travel_speed_log_kmh"]) is not None
    ]

    current_interval = current_features["hours_since_last_transaction_log"]
    current_speed = current_features["travel_speed_log_kmh"]
    signals = {
        "amount_increase": _upper_directional_risk(
            amount_history,
            current_features["amount_log_reais"],
        ),
        "shorter_transaction_interval": (
            _lower_directional_risk(interval_history, current_interval)
            if interval_history and current_interval is not None
            else 0.0
        ),
        "higher_travel_speed": (
            _upper_directional_risk(speed_history, current_speed)
            if speed_history and current_speed is not None
            else 0.0
        ),
        "new_device": _NEW_DEVICE_RISK if is_new_device else 0.0,
    }
    return DirectionalRiskScore(_combine_directional_risks(signals), signals)


def evaluate_payer_behavior(
    history: list[BehaviorObservation],
    current: BehaviorObservation,
) -> DirectionalRiskScore | None:
    if len(history) < MIN_PAYER_HISTORY:
        return None

    ordered_history = sorted(history, key=lambda item: item.transaction_date)[
        -MAX_PAYER_HISTORY:
    ]
    history_features = [
        _features_for_observation(observation, ordered_history[:index])
        for index, observation in enumerate(ordered_history)
    ]
    current_features = _features_for_observation(current, ordered_history)
    known_devices = {observation.device_id for observation in ordered_history}
    return _score_directional_behavior(
        history_features,
        current_features,
        is_new_device=current.device_id not in known_devices,
    )


def score_payer_behavior(
    history: list[BehaviorObservation],
    current: BehaviorObservation,
) -> float | None:
    result = evaluate_payer_behavior(history, current)
    return result.score if result is not None else None


def evaluate_global_behavior(
    history: list[tuple[str, BehaviorObservation]],
    current_payer: str,
    current: BehaviorObservation,
    current_payer_history: list[BehaviorObservation] | None = None,
) -> DirectionalRiskScore | None:
    if len(history) < MIN_GLOBAL_HISTORY:
        return None

    payer_histories: dict[str, list[BehaviorObservation]] = {}
    history_features = []
    for payer, observation in sorted(
        history, key=lambda item: item[1].transaction_date
    ):
        payer_history = payer_histories.setdefault(payer, [])
        history_features.append(
            _features_for_observation(observation, payer_history[-MAX_PAYER_HISTORY:])
        )
        payer_history.append(observation)

    current_history = current_payer_history
    if current_history is None:
        current_history = payer_histories.get(current_payer, [])
    current_history = sorted(current_history, key=lambda item: item.transaction_date)[
        -MAX_PAYER_HISTORY:
    ]
    current_features = _features_for_observation(current, current_history)
    known_devices = {observation.device_id for observation in current_history}
    return _score_directional_behavior(
        history_features,
        current_features,
        is_new_device=(
            bool(current_history) and current.device_id not in known_devices
        ),
    )


def score_global_behavior(
    history: list[tuple[str, BehaviorObservation]],
    current_payer: str,
    current: BehaviorObservation,
) -> float | None:
    result = evaluate_global_behavior(history, current_payer, current)
    return result.score if result is not None else None
