from datetime import datetime, timedelta, timezone

from tests.utils import PayloadGenerator, RequestGenerator


class TestPixTransaction:
    def test_approves_and_persists_analysis(self):
        payload = PayloadGenerator.create_pix_transaction_payload()

        status, response = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        assert set(response) == {"key", "status", "metadata"}
        assert response["status"] == "approved"
        assert response["metadata"]["reasons"] == ["no_risk_rules_triggered"]

        status, stored_analysis = RequestGenerator.GET_pix_transaction(payload["id"])

        assert status == 200
        assert stored_analysis == response

    def test_requires_two_factor_for_high_amount_and_new_device(self):
        baseline = PayloadGenerator.create_pix_transaction_payload()
        status, baseline_response = RequestGenerator.POST_pix_transaction(baseline)
        assert status == 200
        assert baseline_response["status"] == "approved"

        new_device = PayloadGenerator.create_pix_transaction_payload(
            amount=100_000,
            source_document_number=baseline["source_account"]["owner"]["document_number"],
            destination_account_number=baseline["destination_account"]["account_number"],
            device_id="dvc-new-device",
        )
        status, response = RequestGenerator.POST_pix_transaction(new_device)

        assert status == 200
        assert response["status"] == "two_factor_required"
        assert "amount_requires_confirmation" in response["metadata"]["reasons"]
        assert "new_device_for_payer" in response["metadata"]["reasons"]

    def test_requires_two_factor_for_recent_destination_account(self):
        payload = PayloadGenerator.create_pix_transaction_payload(
            destination_opening_date=datetime.now(timezone.utc) - timedelta(days=1),
        )

        status, response = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        assert response["status"] == "two_factor_required"
        assert "destination_account_is_new" in response["metadata"]["reasons"]

    def test_rejects_amount_over_transaction_limit(self):
        payload = PayloadGenerator.create_pix_transaction_payload(amount=500_001)

        status, response = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        assert response["status"] == "rejected"
        assert "amount_over_transaction_limit" in response["metadata"]["reasons"]

    def test_rejects_amount_over_24_hour_limit(self):
        first = PayloadGenerator.create_pix_transaction_payload(amount=99_999)
        status, response = RequestGenerator.POST_pix_transaction(first)
        assert status == 200
        assert response["status"] == "approved"

        for _ in range(2):
            next_transaction = PayloadGenerator.create_pix_transaction_payload(
                amount=99_999,
                source_document_number=first["source_account"]["owner"]["document_number"],
                destination_account_number=first["destination_account"]["account_number"],
                device_id=first["source"]["device_id"],
            )
            status, response = RequestGenerator.POST_pix_transaction(next_transaction)
            assert status == 200
            assert response["status"] == "approved"

        final_transaction = PayloadGenerator.create_pix_transaction_payload(
            amount=99_999,
            source_document_number=first["source_account"]["owner"]["document_number"],
            destination_account_number=first["destination_account"]["account_number"],
            device_id=first["source"]["device_id"],
        )
        status, response = RequestGenerator.POST_pix_transaction(final_transaction)

        assert status == 200
        assert response["status"] == "rejected"
        assert "amount_over_24_hour_limit" in response["metadata"]["reasons"]

    def test_requires_two_factor_after_three_transactions_in_ten_minutes(self):
        now = datetime.now(timezone.utc)
        first = PayloadGenerator.create_pix_transaction_payload(
            transaction_date=now - timedelta(minutes=9),
        )
        for minutes_ago in (9, 6, 3):
            payload = PayloadGenerator.create_pix_transaction_payload(
                transaction_date=now - timedelta(minutes=minutes_ago),
                source_document_number=first["source_account"]["owner"]["document_number"],
                destination_account_number=first["destination_account"]["account_number"],
                device_id=first["source"]["device_id"],
            )
            status, response = RequestGenerator.POST_pix_transaction(payload)
            assert status == 200
            assert response["status"] == "approved"

        fourth = PayloadGenerator.create_pix_transaction_payload(
            transaction_date=now,
            source_document_number=first["source_account"]["owner"]["document_number"],
            destination_account_number=first["destination_account"]["account_number"],
            device_id=first["source"]["device_id"],
        )
        status, response = RequestGenerator.POST_pix_transaction(fourth)

        assert status == 200
        assert response["status"] == "two_factor_required"
        assert "high_transaction_velocity" in response["metadata"]["reasons"]

    def test_validates_payload_and_prevents_duplicate_transaction_ids(self):
        payload = PayloadGenerator.create_pix_transaction_payload()
        payload["unknown"] = "not allowed"

        status, response = RequestGenerator.POST_pix_transaction(payload)
        assert status == 400
        assert response["code"] == "QIT000001"

        payload.pop("unknown")
        status, _ = RequestGenerator.POST_pix_transaction(payload)
        assert status == 200

        status, response = RequestGenerator.POST_pix_transaction(payload)
        assert status == 409
        assert response["code"] == "QIT001009"
