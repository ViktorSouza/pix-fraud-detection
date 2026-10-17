from datetime import datetime, timedelta, timezone
from uuid import uuid4

from tests.utils.random_generator import RandomGenerator


class PayloadGenerator:
    @staticmethod
    def create_pix_transaction_payload(
        amount: int = 25_000,
        transaction_date: datetime = None,
        source_document_number: str = None,
        destination_account_number: str = None,
        device_id: str = None,
        destination_opening_date: datetime = None,
        transaction_id: str = None,
    ) -> dict:
        if transaction_date is None:
            transaction_date = datetime.now(timezone.utc)
        if source_document_number is None:
            digits = f"{uuid4().int % 100_000_000_000:011d}"
            source_document_number = f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"
        if destination_account_number is None:
            destination_account_number = str(uuid4().int)[:12]
        if device_id is None:
            device_id = f"dvc-{uuid4()}"
        if destination_opening_date is None:
            destination_opening_date = transaction_date - timedelta(days=365)
        if transaction_id is None:
            transaction_id = f"pix-{uuid4()}"

        opening_date = transaction_date - timedelta(days=365)
        return {
            "id": transaction_id,
            "amount": amount,
            "transaction_date": transaction_date.isoformat(),
            "source_account": {
                "branch": "0001",
                "account_number": "88213",
                "opening_date": opening_date.isoformat(),
                "owner": {"document_number": source_document_number},
            },
            "destination_account": {
                "participant": "60701190",
                "branch": "3675",
                "account_number": destination_account_number,
                "opening_date": destination_opening_date.isoformat(),
                "owner": {"document_number": "987.654.321-00"},
            },
            "source": {
                "device_id": device_id,
                "channel": "app",
                "ip": "198.51.100.23",
                "geolocation": {"latitude": -23.56, "longitude": -46.65},
            },
        }

    @staticmethod
    def create_sample_entity_payload(
        hello: str = None,
        name: str = None,
        email: str = None,
        document_number: str = None,
        birthdate: str = None,
    ) -> dict:
        """Um cadastro valido, com qualquer campo trocado a pedido.

        Sem argumento nenhum o payload sai aleatorio no que precisa ser
        unico (e-mail e CPF), pra que dois cadastros seguidos nao batam
        na regra de duplicidade. Quem testa FILTRO precisa do contrario
        disso: um valor conhecido, pra poder procurar por ele depois.
        """
        if hello is None:
            hello = "world"

        if name is None:
            name = "Maria da Silva"

        if email is None:
            email = f"maria.silva.{uuid4()}@exemplo.com.br"

        if document_number is None:
            document_number = RandomGenerator.generate_cpf()

        if birthdate is None:
            birthdate = "1990-05-17"

        payload = {
            "hello": hello,
            "name": name,
            "email": email,
            "document_number": document_number,
            "birthdate": birthdate,
        }
        return payload

    @staticmethod
    def create_new_status_payload(new_status: str = None) -> dict:
        payload = {"status": new_status}
        return payload
