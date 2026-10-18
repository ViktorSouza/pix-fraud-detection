from fastapi import status as http_status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from controllers import PixTransactionController
from utils.schema_handler import SchemaHandler


class PixTransactionResource:
    @SchemaHandler.validate("post_pix_transaction.json")
    def on_post(self, payload: dict) -> JSONResponse:
        controller = PixTransactionController()
        analysis = controller.analyze(payload)
        return JSONResponse(
            content=jsonable_encoder(analysis),
            status_code=http_status.HTTP_200_OK,
        )

    def on_get_by_id(self, transaction_id: str) -> JSONResponse:
        controller = PixTransactionController()
        analysis = controller.get_by_id(transaction_id)
        return JSONResponse(
            content=jsonable_encoder(analysis),
            status_code=http_status.HTTP_200_OK,
        )

    def on_post_verify_two_factor(self, transaction_id: str) -> JSONResponse:
        controller = PixTransactionController()
        analysis = controller.mark_two_factor_verified(transaction_id)
        return JSONResponse(
            content=jsonable_encoder(analysis),
            status_code=http_status.HTTP_200_OK,
        )
