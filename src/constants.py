import os
from pathlib import Path


SERVICE_ROOT = os.path.abspath(os.path.dirname(__file__))

# Onde moram os arquivos de schema — os .json que descrevem o formato
# que cada requisicao precisa ter. Quem le essa pasta e o
# src/utils/schema_handler.py.
SCHEMA_PATH = os.path.join(SERVICE_ROOT, "schemas")

APP_ENV = os.environ.get("APP_ENV", "local")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "bootcamp-api")

DATABASE_URL = os.environ.get("DATABASE_URL")
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN")

# A API de boletos: o serviço de fora que este projeto chama pra emitir
# uma cobrança (veja src/connectors/). O endereço vem do ambiente, como
# tudo aqui — na sua máquina ele aponta pro mock server do sábado 4; em
# produção, apontaria pro serviço de verdade. O código não sabe a
# diferença, e esse é o ponto.
BANKSLIP_API_URL = os.environ.get("BANKSLIP_API_URL", "http://localhost:8080")
BANKSLIP_API_INTERNAL_TOKEN = os.environ.get("BANKSLIP_API_INTERNAL_TOKEN", "default_token")

# Quantos segundos esperar pelo serviço de boletos antes de desistir.
# Todo connector TEM um timeout — o porquê está em
# src/connectors/rest_connector.py.
BANKSLIP_API_TIMEOUT = int(os.environ.get("BANKSLIP_API_TIMEOUT", "5"))

PIX_FRAUD_MODEL_PATH = os.environ.get(
    "PIX_FRAUD_MODEL_PATH",
    str(Path(SERVICE_ROOT) / "artifacts" / "pix_fraud_model.joblib"),
)
PIX_TWO_FACTOR_THRESHOLD = float(os.environ.get("PIX_TWO_FACTOR_THRESHOLD", "0.30"))
PIX_REJECT_THRESHOLD = float(os.environ.get("PIX_REJECT_THRESHOLD", "0.70"))

# Rotas públicas: não exigem o header INTERNAL-TOKEN. São as rotas que
# precisam responder pra quem ainda não tem token nenhum: a raiz, que
# diz quem é este serviço, o health check do Docker e a página local de
# testes, que pede o token antes de chamar os endpoints protegidos.
BYPASS_ENDPOINTS = [
    "/",
    "/health_check",
    "/playground",
]

REQUIRED_VARIABLES = ["DATABASE_URL", "INTERNAL_TOKEN"]


def check_variables():
    if not 0 < PIX_TWO_FACTOR_THRESHOLD < PIX_REJECT_THRESHOLD < 1:
        raise EnvironmentError(
            "PIX_TWO_FACTOR_THRESHOLD e PIX_REJECT_THRESHOLD devem satisfazer "
            "0 < two-factor < rejection < 1."
        )

    missing = []
    for name in REQUIRED_VARIABLES:
        if not globals().get(name):
            missing.append(name)

    if missing:
        raise EnvironmentError(
            f"Faltam variáveis de ambiente: {', '.join(missing)}. "
            "Rodando com 'docker compose up' elas já vêm preenchidas. "
            "Fora do Docker, copie o .env.example para .env."
        )
