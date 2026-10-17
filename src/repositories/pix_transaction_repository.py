from datetime import datetime
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import Float, case, func

from database import Context
from models import PixTransaction


BEHAVIOR_HISTORY_LIMIT = 200


class PixTransactionRepository:
    def __init__(self, context: Context) -> None:
        self.session = context.db_session

    def get_by_transaction_id(self, transaction_id: str) -> PixTransaction:
        return (
            self.session.query(PixTransaction)
            .filter(PixTransaction.transaction_id == transaction_id)
            .first()
        )

    def get_approved_profile(
        self,
        source_document_number: str,
        since: datetime,
        velocity_since: datetime,
        transaction_date: datetime,
        destination_fingerprint: str,
        device_id: str,
    ) -> tuple[int, int, int, bool, bool]:
        profile = (
            self.session.query(
                func.count(PixTransaction.id),
                func.coalesce(func.sum(PixTransaction.amount), 0),
                func.coalesce(
                    func.sum(
                        case(
                            (PixTransaction.transaction_date >= velocity_since, 1),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.coalesce(
                    func.bool_or(PixTransaction.destination_fingerprint == destination_fingerprint),
                    False,
                ),
                func.coalesce(func.bool_or(PixTransaction.device_id == device_id), False),
            )
            .filter(
                PixTransaction.source_document_number == source_document_number,
                PixTransaction.status == "approved",
                PixTransaction.transaction_date >= since,
                PixTransaction.transaction_date < transaction_date,
            )
            .one()
        )

        count, amount, velocity_count, destination_seen, device_seen = profile
        return count, amount, velocity_count, destination_seen, device_seen

    def get_latest_approved_location(
        self,
        source_document_number: str,
        since: datetime,
        transaction_date: datetime,
    ) -> tuple[datetime, float, float] | None:
        latitude = PixTransaction.transaction_data["source"]["geolocation"]["latitude"].astext
        longitude = PixTransaction.transaction_data["source"]["geolocation"]["longitude"].astext
        location = (
            self.session.query(
                PixTransaction.transaction_date,
                latitude.cast(Float),
                longitude.cast(Float),
            )
            .filter(
                PixTransaction.source_document_number == source_document_number,
                PixTransaction.status == "approved",
                PixTransaction.transaction_date >= since,
                PixTransaction.transaction_date < transaction_date,
            )
            .order_by(PixTransaction.transaction_date.desc(), PixTransaction.id.desc())
            .first()
        )

        if location is None or location[1] is None or location[2] is None:
            return None
        return location[0], float(location[1]), float(location[2])

    def get_approved_behavior_history(
        self,
        source_document_number: str,
        transaction_date: datetime,
    ) -> list[tuple[int, datetime, str, float | None, float | None]]:
        return self._get_approved_behavior_rows(
            source_document_number=source_document_number,
            transaction_date=transaction_date,
        )

    def get_global_approved_behavior_history(
        self,
        transaction_date: datetime,
        limit: int = BEHAVIOR_HISTORY_LIMIT,
    ) -> list[tuple[str, int, datetime, str, float | None, float | None]]:
        latitude = PixTransaction.transaction_data["source"]["geolocation"]["latitude"].astext
        longitude = PixTransaction.transaction_data["source"]["geolocation"]["longitude"].astext
        rows = (
            self.session.query(
                PixTransaction.source_document_number,
                PixTransaction.amount,
                PixTransaction.transaction_date,
                PixTransaction.device_id,
                latitude.cast(Float),
                longitude.cast(Float),
            )
            .filter(
                PixTransaction.status == "approved",
                PixTransaction.transaction_date < transaction_date,
            )
            .order_by(PixTransaction.transaction_date.desc(), PixTransaction.id.desc())
            .limit(limit)
            .all()
        )
        return [
            (payer, amount, date, device, lat, lon)
            for payer, amount, date, device, lat, lon in reversed(rows)
        ]

    def _get_approved_behavior_rows(
        self,
        source_document_number: str,
        transaction_date: datetime,
    ) -> list[tuple[int, datetime, str, float | None, float | None]]:
        latitude = PixTransaction.transaction_data["source"]["geolocation"]["latitude"].astext
        longitude = PixTransaction.transaction_data["source"]["geolocation"]["longitude"].astext
        rows = (
            self.session.query(
                PixTransaction.amount,
                PixTransaction.transaction_date,
                PixTransaction.device_id,
                latitude.cast(Float),
                longitude.cast(Float),
            )
            .filter(
                PixTransaction.source_document_number == source_document_number,
                PixTransaction.status == "approved",
                PixTransaction.transaction_date < transaction_date,
            )
            .order_by(PixTransaction.transaction_date.desc(), PixTransaction.id.desc())
            .limit(BEHAVIOR_HISTORY_LIMIT)
            .all()
        )
        return [
            (amount, date, device, lat, lon)
            for amount, date, device, lat, lon in reversed(rows)
        ]

    def create(
        self,
        transaction_data: dict,
        transaction_date: datetime,
        source_document_number: str,
        destination_fingerprint: str,
        status: str,
        metadata: dict,
    ) -> PixTransaction:
        pix_transaction = PixTransaction()
        pix_transaction.transaction_id = transaction_data["id"]
        pix_transaction.analysis_key = str(uuid4())
        pix_transaction.amount = transaction_data["amount"]
        pix_transaction.transaction_date = transaction_date
        pix_transaction.source_document_number = source_document_number
        pix_transaction.destination_fingerprint = destination_fingerprint
        pix_transaction.device_id = transaction_data["source"]["device_id"]
        pix_transaction.status = status
        pix_transaction.transaction_data = transaction_data
        pix_transaction.analysis_metadata = metadata

        self.session.add(pix_transaction)
        return pix_transaction

    @staticmethod
    def fingerprint_destination(destination_account: dict) -> str:
        identity = ":".join(
            (
                destination_account["participant"],
                destination_account["branch"],
                destination_account["account_number"],
            )
        )
        return sha256(identity.encode("utf-8")).hexdigest()
