from sqlalchemy import BigInteger, CHAR, Column, DateTime, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from models.base import Base


class PixTransaction(Base):
    __tablename__ = "pix_transaction"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(255), nullable=False)
    analysis_key = Column(CHAR(36), nullable=False)
    amount = Column(BigInteger, nullable=False)
    transaction_date = Column(DateTime(timezone=True), nullable=False)
    source_document_number = Column(String(18), nullable=False)
    destination_fingerprint = Column(CHAR(64), nullable=False)
    device_id = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False)
    transaction_data = Column(JSONB, nullable=False)
    analysis_metadata = Column("metadata", JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("transaction_id"),
        UniqueConstraint("analysis_key"),
        Index("ix_pix_transaction_source_date_status", "source_document_number", "transaction_date", "status"),
    )
