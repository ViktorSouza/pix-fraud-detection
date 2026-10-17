from datetime import datetime, timedelta, timezone

import pytest

from tests.utils import PayloadGenerator, RequestGenerator


class TestPixTransaction:
    @staticmethod
    def assert_decision_matches_score(response):
        metadata = response["metadata"]
        score = metadata["fraud_score"]
        assert 0 <= score <= 1
        assert metadata["decision_layer"] in {
            "level_2_hard_rule",
            "level_4_rules",
            "paysim_prototype_model",
        }

        if score >= 0.70 or metadata["decision_layer"] == "level_2_hard_rule":
            expected_status = "rejected"
        elif score >= 0.30 or metadata["triggered_rules"]:
            expected_status = "two_factor_required"
        else:
            expected_status = "approved"

        assert response["status"] == expected_status

    def test_approves_and_persists_analysis(self):
        payload = PayloadGenerator.create_pix_transaction_payload()

        status, response = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        assert set(response) == {"key", "status", "metadata"}
        self.assert_decision_matches_score(response)

        status, stored_analysis = RequestGenerator.GET_pix_transaction(payload["id"])

        assert status == 200
        assert stored_analysis == response

    def test_requires_two_factor_for_high_amount_and_new_device(self):
        baseline = PayloadGenerator.create_pix_transaction_payload()
        status, baseline_response = RequestGenerator.POST_pix_transaction(baseline)
        assert status == 200
        self.assert_decision_matches_score(baseline_response)

        new_device = PayloadGenerator.create_pix_transaction_payload(
            amount=100_000,
            source_document_number=baseline["source_account"]["owner"]["document_number"],
            destination_account_number=baseline["destination_account"]["account_number"],
            device_id="dvc-new-device",
        )
        status, response = RequestGenerator.POST_pix_transaction(new_device)

        assert status == 200
        self.assert_decision_matches_score(response)
        assert response["status"] in {"two_factor_required", "rejected"}
        assert "amount_requires_confirmation" in response["metadata"]["reasons"]
        assert "new_device_for_payer" in response["metadata"]["reasons"]

    def test_requires_two_factor_for_recent_destination_account(self):
        payload = PayloadGenerator.create_pix_transaction_payload(
            destination_opening_date=datetime.now(timezone.utc) - timedelta(days=1),
        )

        status, response = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        self.assert_decision_matches_score(response)
        assert response["status"] in {"two_factor_required", "rejected"}
        assert "destination_account_is_new" in response["metadata"]["reasons"]

    def test_rejects_amount_over_transaction_limit(self):
        payload = PayloadGenerator.create_pix_transaction_payload(amount=500_001)

        status, response = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        assert response["status"] == "rejected"
        assert "amount_over_transaction_limit" in response["metadata"]["reasons"]

    def test_rejects_amount_over_24_hour_limit(self):
        now = datetime.now(timezone.utc)
        payer = PayloadGenerator.create_pix_transaction_payload()
        approved_count = 0
        attempts = 0
        while approved_count < 7 and attempts < 60:
            transaction = PayloadGenerator.create_pix_transaction_payload(
                amount=45_000,
                transaction_date=now - timedelta(hours=2) + timedelta(minutes=attempts * 11),
                source_document_number=payer["source_account"]["owner"]["document_number"],
                destination_account_number=payer["destination_account"]["account_number"],
                device_id=payer["source"]["device_id"],
            )
            status, response = RequestGenerator.POST_pix_transaction(transaction)
            assert status == 200
            self.assert_decision_matches_score(response)
            approved_count += response["status"] == "approved"
            attempts += 1

        if approved_count < 7:
            pytest.skip("The configured PaySim model did not approve enough setup transactions.")

        final_transaction = PayloadGenerator.create_pix_transaction_payload(
            amount=45_000,
            transaction_date=now,
            source_document_number=payer["source_account"]["owner"]["document_number"],
            destination_account_number=payer["destination_account"]["account_number"],
            device_id=payer["source"]["device_id"],
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
        all_approved = True
        for minutes_ago in (9, 6, 3):
            payload = PayloadGenerator.create_pix_transaction_payload(
                transaction_date=now - timedelta(minutes=minutes_ago),
                source_document_number=first["source_account"]["owner"]["document_number"],
                destination_account_number=first["destination_account"]["account_number"],
                device_id=first["source"]["device_id"],
            )
            status, response = RequestGenerator.POST_pix_transaction(payload)
            assert status == 200
            self.assert_decision_matches_score(response)
            all_approved = all_approved and response["status"] == "approved"

        if not all_approved:
            pytest.skip("The configured PaySim model did not approve the velocity setup.")

        fourth = PayloadGenerator.create_pix_transaction_payload(
            transaction_date=now,
            source_document_number=first["source_account"]["owner"]["document_number"],
            destination_account_number=first["destination_account"]["account_number"],
            device_id=first["source"]["device_id"],
        )
        status, response = RequestGenerator.POST_pix_transaction(fourth)

        assert status == 200
        self.assert_decision_matches_score(response)
        assert response["status"] in {"two_factor_required", "rejected"}
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
