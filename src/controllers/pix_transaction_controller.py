from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import IntegrityError

from controllers.base_controller import BaseController
from dtos import PixTransactionDTO
from errors import (
    DuplicatedPixTransaction,
    InvalidPixTimestamp,
    NotFoundPixTransaction,
)
from repositories import PixTransactionRepository

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
        self._parse_timestamp(source_account["opening_date"])
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

        if rejection_reasons:
            status = "rejected"
        elif two_factor_reasons:
            status = "two_factor_required"
        else:
            status = "approved"

        reasons = rejection_reasons + two_factor_reasons
        if not reasons:
            reasons.append("no_risk_rules_triggered")

        metadata = {
            "reasons": reasons,
        }
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
