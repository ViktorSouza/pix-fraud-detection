from datetime import datetime
from math import asin, cos, radians, sin, sqrt


EARTH_RADIUS_KM = 6371.0088


def calculate_location_speed_kmh(
    previous_latitude: float,
    previous_longitude: float,
    previous_date: datetime,
    current_latitude: float,
    current_longitude: float,
    current_date: datetime,
) -> tuple[float, float] | None:
    elapsed_hours = (current_date - previous_date).total_seconds() / 3600
    if elapsed_hours <= 0:
        return None

    latitude_delta = radians(current_latitude - previous_latitude)
    longitude_delta = radians(current_longitude - previous_longitude)
    previous_latitude_radians = radians(previous_latitude)
    current_latitude_radians = radians(current_latitude)

    haversine = (
        sin(latitude_delta / 2) ** 2
        + cos(previous_latitude_radians)
        * cos(current_latitude_radians)
        * sin(longitude_delta / 2) ** 2
    )
    distance_km = 2 * EARTH_RADIUS_KM * asin(sqrt(min(1.0, haversine)))
    speed_kmh = distance_km / elapsed_hours
    return distance_km, speed_kmh
