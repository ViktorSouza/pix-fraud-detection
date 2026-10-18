from errors import QIException


class NotFoundSampleEntity(QIException):
    code = "QIT001001"

    def __init__(self, sample_entity_key) -> None:
        title = "Entity not Found"
        http_status = 404
        description = f"Entity with key {sample_entity_key} was not found."
        translation = f"A entidade com chave {sample_entity_key} não foi encontrada."
        super().__init__(title, self.code, http_status, description, translation)


class SampleEntityFinalStatus(QIException):
    code = "QIT001002"

    def __init__(self, old_status, new_status) -> None:
        title = "Entity cannot change status"
        http_status = 409
        description = f"Entity with status {old_status} cannot update to {new_status}."
        translation = "Essa entidade não pode ser atualizada."
        super().__init__(title, self.code, http_status, description, translation)


class InvalidDocumentNumber(QIException):
    """O CPF tem o formato certo e não existe.

    422, e não 400, de propósito: 400 quer dizer "não consegui ler o seu
    pedido". Aqui a API leu, entendeu, e o valor é que não pode existir —
    os dois últimos dígitos não batem com a conta. A diferença está
    explicada em src/utils/document_number.py.
    """

    code = "QIT001003"

    def __init__(self, document_number) -> None:
        title = "Invalid Document Number"
        http_status = 422
        description = f"The document number {document_number} is not a valid CPF."
        translation = "O CPF informado não é válido."
        super().__init__(title, self.code, http_status, description, translation)


class DuplicatedDocumentNumber(QIException):
    """Já existe um cadastro com este CPF.

    409 Conflict: o pedido está correto em si, e o que impede é o que já
    está no banco. É a mesma família do SampleEntityFinalStatus aqui em
    cima — conflito com o que já existe, não erro de quem pediu.
    """

    code = "QIT001004"

    def __init__(self, document_number) -> None:
        title = "Document Number already registered"
        http_status = 409
        description = f"There is already an entity with the document number {document_number}."
        translation = "Já existe um cadastro com este CPF."
        super().__init__(title, self.code, http_status, description, translation)


class DuplicatedEmail(QIException):
    code = "QIT001005"

    def __init__(self, email) -> None:
        title = "Email already registered"
        http_status = 409
        description = f"There is already an entity with the email {email}."
        translation = "Já existe um cadastro com este e-mail."
        super().__init__(title, self.code, http_status, description, translation)


class UnderageSampleEntity(QIException):
    code = "QIT001006"

    def __init__(self, age, minimum_age) -> None:
        title = "Entity is underage"
        http_status = 422
        description = f"The entity is {age} years old, and the minimum is {minimum_age}."
        translation = f"É preciso ter pelo menos {minimum_age} anos."
        super().__init__(title, self.code, http_status, description, translation)


class InvalidBirthdate(QIException):
    """A data tem o formato certo e não existe no calendário.

    Existe porque o `pattern` do schema sabe contar dígitos, não dias:
    "2025-02-30" e "9999-99-99" passam pelo regex e morrem no
    `date.fromisoformat`. Sem esta classe, esse ValueError virava 500 —
    a API culpando a si mesma por um erro de quem chamou.
    """

    code = "QIT001007"

    def __init__(self, birthdate) -> None:
        title = "Invalid Birthdate"
        http_status = 422
        description = f"The birthdate {birthdate} is not a real date."
        translation = "A data de nascimento informada não existe."
        super().__init__(title, self.code, http_status, description, translation)


class NotFoundPixTransaction(QIException):
    code = "QIT001008"

    def __init__(self, transaction_id) -> None:
        title = "Pix transaction not found"
        http_status = 404
        description = f"Pix transaction {transaction_id} was not found."
        translation = "A transação Pix não foi encontrada."
        super().__init__(title, self.code, http_status, description, translation)


class DuplicatedPixTransaction(QIException):
    code = "QIT001009"

    def __init__(self, transaction_id) -> None:
        title = "Pix transaction already analyzed"
        http_status = 409
        description = f"Pix transaction {transaction_id} has already been analyzed."
        translation = "Esta transação Pix já foi analisada."
        super().__init__(title, self.code, http_status, description, translation)


class PixTwoFactorNotRequired(QIException):
    code = "QIT001012"

    def __init__(self, transaction_id) -> None:
        title = "Pix transaction does not require two-factor verification"
        http_status = 409
        description = (
            f"Pix transaction {transaction_id} is not waiting for two-factor verification."
        )
        translation = "Esta transação Pix não está aguardando verificação de duas etapas."
        super().__init__(title, self.code, http_status, description, translation)


class PixTwoFactorSimulationUnavailable(QIException):
    code = "QIT001013"

    def __init__(self) -> None:
        title = "Two-factor simulation unavailable"
        http_status = 403
        description = "Simulated two-factor verification is only available in development environments."
        translation = "A simulação da verificação em duas etapas só está disponível em desenvolvimento."
        super().__init__(title, self.code, http_status, description, translation)


class InvalidPixTimestamp(QIException):
    code = "QIT001010"

    def __init__(self, timestamp) -> None:
        title = "Invalid Pix timestamp"
        http_status = 422
        description = f"The timestamp {timestamp} must be a valid date and time with a timezone."
        translation = "A data e hora da transação devem incluir um fuso horário válido."
        super().__init__(title, self.code, http_status, description, translation)


class FraudModelUnavailable(QIException):
    code = "QIT001011"

    def __init__(self, model_path) -> None:
        title = "Fraud model unavailable"
        http_status = 503
        description = f"The fraud model artifact was not found at {model_path}."
        translation = "O modelo antifraude ainda não foi treinado ou configurado."
        super().__init__(title, self.code, http_status, description, translation)
