from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from constants import (
    PIX_FRAUD_MODEL_PATH,
    PIX_MAX_LOCATION_SPEED_KMH,
    PIX_REJECT_THRESHOLD,
    PIX_TWO_FACTOR_THRESHOLD,
)
from controllers.base_controller import BaseController
from dtos import PixTransactionDTO
from errors import (
    DuplicatedPixTransaction,
    InvalidPixTimestamp,
    NotFoundPixTransaction,
)
from fraud_model import build_pix_model_features, combine_decisions, score_transaction
from repositories import PixTransactionRepository
from utils.geolocation import calculate_location_speed_kmh

MAX_TRANSACTION_AMOUNT_CENTS = 500_000
MAX_APPROVED_AMOUNT_PER_24_HOURS_CENTS = 300_000
TWO_FACTOR_AMOUNT_CENTS = 100_000
NEW_DESTINATION_AGE_DAYS = 7
VELOCITY_WINDOW_MINUTES = 10
VELOCITY_TRANSACTION_LIMIT = 3


class PixTransactionController(BaseController):
    """Avalia regras antifraude locais e registra todas as análises."""

    def __init__(self) -> None:
        super().__init__(__name__)
        self.pix_transaction_repository = PixTransactionRepository(self.context)

    def analyze(self, transaction_data: dict) -> dict:
        transaction_id = transaction_data["id"]
        if self.pix_transaction_repository.get_by_transaction_id(transaction_id) is not None:
            raise DuplicatedPixTransaction(transaction_id)

        transaction_date = self._parse_timestamp(transaction_data["transaction_date"])
        source_account = transaction_data["source_account"]
        destination_account = transaction_data["destination_account"]
        source_document_number = source_account["owner"]["document_number"]
        source_account_opening_date = self._parse_timestamp(source_account["opening_date"])
        destination_fingerprint = self.pix_transaction_repository.fingerprint_destination(
            destination_account
        )
        device_id = transaction_data["source"]["device_id"]
        destination_opening_date = self._parse_timestamp(destination_account["opening_date"])

        profile = self.pix_transaction_repository.get_approved_profile(
            source_document_number=source_document_number,
            since=transaction_date - timedelta(hours=24),
            velocity_since=transaction_date - timedelta(minutes=VELOCITY_WINDOW_MINUTES),
            transaction_date=transaction_date,
            destination_fingerprint=destination_fingerprint,
            device_id=device_id,
        )
        previous_count, previous_amount, velocity_count, destination_seen, device_seen = profile

        rejection_reasons = []
        two_factor_reasons = []
        location_velocity = None

        if transaction_data["amount"] > MAX_TRANSACTION_AMOUNT_CENTS:
            rejection_reasons.append("amount_over_transaction_limit")

        if previous_amount + transaction_data["amount"] > MAX_APPROVED_AMOUNT_PER_24_HOURS_CENTS:
            rejection_reasons.append("amount_over_24_hour_limit")

        if transaction_data["amount"] >= TWO_FACTOR_AMOUNT_CENTS:
            two_factor_reasons.append("amount_requires_confirmation")

        if transaction_date - destination_opening_date < timedelta(days=NEW_DESTINATION_AGE_DAYS):
            two_factor_reasons.append("destination_account_is_new")

        if velocity_count >= VELOCITY_TRANSACTION_LIMIT:
            two_factor_reasons.append("high_transaction_velocity")

        if previous_count and not destination_seen:
            two_factor_reasons.append("new_destination_for_payer")

        if previous_count and not device_seen:
            two_factor_reasons.append("new_device_for_payer")

        current_location = transaction_data["source"].get("geolocation")
        if current_location is not None:
            previous_location = self.pix_transaction_repository.get_latest_approved_location(
                source_document_number=source_document_number,
                since=transaction_date - timedelta(hours=24),
                transaction_date=transaction_date,
            )
            if previous_location is not None:
                previous_date, previous_latitude, previous_longitude = previous_location
                movement = calculate_location_speed_kmh(
                    previous_latitude=previous_latitude,
                    previous_longitude=previous_longitude,
                    previous_date=previous_date,
                    current_latitude=current_location["latitude"],
                    current_longitude=current_location["longitude"],
                    current_date=transaction_date,
                )
                if movement is not None:
                    distance_km, speed_kmh = movement
                    location_velocity = {
                        "distance_km": round(distance_km, 2),
                        "speed_kmh": round(speed_kmh, 2),
                        "threshold_kmh": PIX_MAX_LOCATION_SPEED_KMH,
                    }
                    if speed_kmh > PIX_MAX_LOCATION_SPEED_KMH:
                        two_factor_reasons.append("impossible_location_velocity")

        model_features = build_pix_model_features(
            amount_cents=transaction_data["amount"],
            transaction_date=transaction_date,
            payer_account_age_days=(
                transaction_date - source_account_opening_date
            ).total_seconds()
            / 86400,
            prior_approved_count_24h=previous_count,
            prior_approved_amount_cents=previous_amount,
            prior_approved_count_10m=velocity_count,
            destination_seen=destination_seen,
            device_seen=device_seen,
            location_speed_kmh=location_velocity["speed_kmh"] if location_velocity else None,
        )
        fraud_score = score_transaction(
            amount_cents=transaction_data["amount"],
            transaction_date=transaction_date,
            model_path=Path(PIX_FRAUD_MODEL_PATH),
            model_features=model_features,
        )

        status, decision_layer, model_reasons = combine_decisions(
            fraud_score=fraud_score,
            rejection_reasons=rejection_reasons,
            two_factor_reasons=two_factor_reasons,
            two_factor_threshold=PIX_TWO_FACTOR_THRESHOLD,
            reject_threshold=PIX_REJECT_THRESHOLD,
        )

        reasons = rejection_reasons + two_factor_reasons + model_reasons
        if not reasons:
            reasons.append("no_risk_rules_triggered")

        metadata = {
            "reasons": reasons,
            "fraud_score": round(fraud_score, 6),
            "decision_layer": decision_layer,
            "triggered_rules": rejection_reasons + two_factor_reasons,
            "model_features": model_features,
        }
        if location_velocity is not None:
            metadata["location_velocity"] = location_velocity

        pix_transaction = self.pix_transaction_repository.create(
            transaction_data=transaction_data,
            transaction_date=transaction_date,
            source_document_number=source_document_number,
            destination_fingerprint=destination_fingerprint,
            status=status,
            metadata=metadata,
        )
        result = PixTransactionDTO.obj_to_dict(pix_transaction)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            if self.pix_transaction_repository.get_by_transaction_id(transaction_id) is not None:
                raise DuplicatedPixTransaction(transaction_id) from None
            raise
        return result

    def get_by_id(self, transaction_id: str) -> dict:
        pix_transaction = self.pix_transaction_repository.get_by_transaction_id(transaction_id)
        if pix_transaction is None:
            raise NotFoundPixTransaction(transaction_id)

        return PixTransactionDTO.obj_to_dict(pix_transaction)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise InvalidPixTimestamp(value) from None

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise InvalidPixTimestamp(value)

        return parsed.astimezone(timezone.utc)
