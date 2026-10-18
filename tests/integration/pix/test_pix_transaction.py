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
            "payer_behavior_anomaly",
            "global_behavior_anomaly",
            "global_classifier_fallback",
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
        features = response["metadata"]["model_features"]
        assert features["prior_approved_count_24h"] == 0
        assert features["has_prior_approved_history"] == 0
        assert features["has_location_velocity"] == 0
        assert response["metadata"]["score_source"] in {
            "payer_approved_history",
            "global_approved_history",
            "paysim_or_labeled_global_model",
        }

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

    def test_development_two_factor_verification_approves_transaction(self):
        payload = PayloadGenerator.create_pix_transaction_payload(amount=100_000)
        status, analysis = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        if analysis["status"] != "two_factor_required":
            pytest.skip("The configured model rejected this verification test transaction.")

        status, verified = RequestGenerator.POST_verify_pix_two_factor(payload["id"])

        assert status == 200
        assert verified["status"] == "approved"
        assert verified["metadata"]["two_factor_verification"]["status"] == "verified"
        assert verified["metadata"]["two_factor_verification"]["method"] == "development_simulation"
        assert verified["metadata"]["two_factor_verification"]["verified_at"]

        status, stored = RequestGenerator.GET_pix_transaction(payload["id"])
        assert status == 200
        assert stored == verified

        status, repeated = RequestGenerator.POST_verify_pix_two_factor(payload["id"])
        assert status == 200
        assert repeated == verified

    def test_recent_two_factor_allows_a_similar_amount_without_repeating_amount_rule(self):
        now = datetime.now(timezone.utc)
        first = PayloadGenerator.create_pix_transaction_payload(
            amount=100_000,
            transaction_date=now,
        )
        status, analysis = RequestGenerator.POST_pix_transaction(first)
        assert status == 200
        if analysis["status"] != "two_factor_required":
            pytest.skip("The configured model did not leave the setup transaction pending 2FA.")

        status, verified = RequestGenerator.POST_verify_pix_two_factor(first["id"])
        assert status == 200
        assert verified["status"] == "approved"

        similar = PayloadGenerator.create_pix_transaction_payload(
            amount=125_000,
            transaction_date=now + timedelta(minutes=1),
            source_document_number=first["source_account"]["owner"]["document_number"],
            destination_account_number=first["destination_account"]["account_number"],
            device_id=first["source"]["device_id"],
        )
        status, response = RequestGenerator.POST_pix_transaction(similar)

        assert status == 200
        assert "amount_requires_confirmation" not in response["metadata"]["reasons"]
        assert response["metadata"]["two_factor_amortization"] == {
            "confirmed_amount_cents": 100_000,
            "window_minutes": 30,
            "amount_tolerance_percent": 25,
            "amount_requires_confirmation_suppressed": True,
            "behavior_amount_increase_suppressed": False,
        }

    def test_two_factor_verification_returns_not_found_for_unknown_transaction(self):
        status, response = RequestGenerator.POST_verify_pix_two_factor(
            f"missing-{datetime.now(timezone.utc).timestamp()}"
        )

        assert status == 404
        assert response["code"] == "QIT001008"

    def test_requires_two_factor_for_impossible_location_velocity(self):
        now = datetime.now(timezone.utc)
        baseline = PayloadGenerator.create_pix_transaction_payload(
            amount=1_000,
            transaction_date=now - timedelta(hours=2),
        )
        baseline["source"]["geolocation"] = {"latitude": 0, "longitude": 0}
        status, baseline_response = RequestGenerator.POST_pix_transaction(baseline)
        assert status == 200

        if baseline_response["status"] != "approved":
            pytest.skip("The configured model did not approve the location history transaction.")

        current = PayloadGenerator.create_pix_transaction_payload(
            transaction_date=now,
            source_document_number=baseline["source_account"]["owner"]["document_number"],
            destination_account_number=baseline["destination_account"]["account_number"],
            device_id=baseline["source"]["device_id"],
        )
        current["source"]["geolocation"] = {"latitude": 0, "longitude": 60}
        status, response = RequestGenerator.POST_pix_transaction(current)

        assert status == 200
        self.assert_decision_matches_score(response)
        assert response["status"] in {"two_factor_required", "rejected"}
        assert "impossible_location_velocity" in response["metadata"]["reasons"]
        assert response["metadata"]["location_velocity"]["speed_kmh"] > 3000

    def test_rejects_amount_over_transaction_limit(self):
        payload = PayloadGenerator.create_pix_transaction_payload(amount=500_001)

        status, response = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        assert response["status"] == "rejected"
        assert "amount_over_transaction_limit" in response["metadata"]["reasons"]

    def test_persists_large_rejected_amounts_that_fit_postgres_bigint(self):
        payload = PayloadGenerator.create_pix_transaction_payload(amount=25_000_000_000)

        status, response = RequestGenerator.POST_pix_transaction(payload)

        assert status == 200
        assert response["status"] == "rejected"
        assert "amount_over_transaction_limit" in response["metadata"]["reasons"]

        status, stored_analysis = RequestGenerator.GET_pix_transaction(payload["id"])

        assert status == 200
        assert stored_analysis == response

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
