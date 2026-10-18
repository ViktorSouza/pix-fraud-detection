from datetime import datetime
from types import SimpleNamespace

import pytest

from controllers.pix_transaction_controller import PixTransactionController
from errors import (
    NotFoundPixTransaction,
    PixTwoFactorNotRequired,
    PixTwoFactorSimulationUnavailable,
)


class FakeSession:
    def __init__(self):
        self.commit_count = 0

    def commit(self):
        self.commit_count += 1


class FakeRepository:
    def __init__(self, transaction):
        self.transaction = transaction

    def get_by_transaction_id(self, transaction_id):
        return self.transaction


def create_controller(transaction):
    controller = object.__new__(PixTransactionController)
    controller.session = FakeSession()
    controller.pix_transaction_repository = FakeRepository(transaction)
    return controller


def create_transaction(status="two_factor_required"):
    return SimpleNamespace(
        analysis_key="analysis-key",
        status=status,
        analysis_metadata={"fraud_score": 0.4},
    )


def test_development_verification_approves_transaction(monkeypatch):
    monkeypatch.setattr("controllers.pix_transaction_controller.PIX_DEV_MODE", True)
    transaction = create_transaction()
    controller = create_controller(transaction)

    result = controller.mark_two_factor_verified("transaction-id")

    verification = result["metadata"]["two_factor_verification"]
    assert verification["status"] == "verified"
    assert verification["method"] == "development_simulation"
    assert datetime.fromisoformat(verification["verified_at"]).utcoffset().total_seconds() == 0
    assert result["status"] == "approved"
    assert controller.session.commit_count == 1


def test_repeated_verification_is_idempotent(monkeypatch):
    monkeypatch.setattr("controllers.pix_transaction_controller.PIX_DEV_MODE", True)
    transaction = create_transaction()
    controller = create_controller(transaction)

    first_result = controller.mark_two_factor_verified("transaction-id")
    second_result = controller.mark_two_factor_verified("transaction-id")

    assert second_result == first_result
    assert controller.session.commit_count == 1


def test_simulated_verification_is_unavailable_outside_development(monkeypatch):
    monkeypatch.setattr("controllers.pix_transaction_controller.PIX_DEV_MODE", False)
    controller = create_controller(create_transaction())

    with pytest.raises(PixTwoFactorSimulationUnavailable):
        controller.mark_two_factor_verified("transaction-id")

    assert controller.session.commit_count == 0


def test_verification_requires_pending_two_factor(monkeypatch):
    monkeypatch.setattr("controllers.pix_transaction_controller.PIX_DEV_MODE", True)
    controller = create_controller(create_transaction(status="approved"))

    with pytest.raises(PixTwoFactorNotRequired):
        controller.mark_two_factor_verified("transaction-id")

    assert controller.session.commit_count == 0


def test_verification_rejects_unknown_transaction(monkeypatch):
    monkeypatch.setattr("controllers.pix_transaction_controller.PIX_DEV_MODE", True)
    controller = create_controller(None)

    with pytest.raises(NotFoundPixTransaction):
        controller.mark_two_factor_verified("missing")

    assert controller.session.commit_count == 0
