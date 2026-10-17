from copy import deepcopy

from models import PixTransaction


class PixTransactionDTO:
    @staticmethod
    def obj_to_dict(pix_transaction: PixTransaction) -> dict:
        return {
            "key": pix_transaction.analysis_key,
            "status": pix_transaction.status,
            "metadata": deepcopy(pix_transaction.analysis_metadata),
        }
