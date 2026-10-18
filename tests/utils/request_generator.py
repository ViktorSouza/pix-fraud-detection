from os import environ

from tests.utils.requisition import ClientRequisition, BaseConnectorResponse


INTERNAL_TOKEN = environ.get("INTERNAL_TOKEN", "default_token")


class RequestGenerator:
    @staticmethod
    def POST_pix_transaction(pix_transaction_payload: dict) -> BaseConnectorResponse:
        response = ClientRequisition.send(
            "POST",
            "/pix/transaction",
            payload=pix_transaction_payload,
            headers={"INTERNAL-TOKEN": INTERNAL_TOKEN},
        )
        return response.response_status, response.response_json

    @staticmethod
    def GET_pix_transaction(transaction_id: str) -> BaseConnectorResponse:
        response = ClientRequisition.send(
            "GET",
            f"/pix/transaction/{transaction_id}",
            headers={"INTERNAL-TOKEN": INTERNAL_TOKEN},
        )
        return response.response_status, response.response_json

    @staticmethod
    def POST_verify_pix_two_factor(transaction_id: str) -> BaseConnectorResponse:
        response = ClientRequisition.send(
            "POST",
            f"/pix/transaction/{transaction_id}/verify-two-factor",
            headers={"INTERNAL-TOKEN": INTERNAL_TOKEN},
        )
        return response.response_status, response.response_json
