from datetime import datetime, timedelta, timezone

import pytest

from utils.geolocation import calculate_location_speed_kmh


def test_calculates_distance_and_speed_between_locations():
    now = datetime.now(timezone.utc)

    movement = calculate_location_speed_kmh(
        previous_latitude=0,
        previous_longitude=0,
        previous_date=now,
        current_latitude=0,
        current_longitude=60,
        current_date=now + timedelta(hours=2),
    )

    assert movement is not None
    distance_km, speed_kmh = movement
    assert distance_km == pytest.approx(6671.7, abs=1)
    assert speed_kmh == pytest.approx(distance_km / 2)


def test_ignores_non_positive_elapsed_time():
    now = datetime.now(timezone.utc)

    movement = calculate_location_speed_kmh(
        previous_latitude=0,
        previous_longitude=0,
        previous_date=now,
        current_latitude=0,
        current_longitude=60,
        current_date=now,
    )

    assert movement is None
