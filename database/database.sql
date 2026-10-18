

CREATE TABLE pix_transaction(
    id                              SERIAL PRIMARY KEY,
    transaction_id                  VARCHAR(255) NOT NULL UNIQUE,
    analysis_key                    CHAR(36) NOT NULL UNIQUE,
    amount                          BIGINT NOT NULL,
    transaction_date                TIMESTAMPTZ NOT NULL,
    source_document_number          VARCHAR(18) NOT NULL,
    destination_fingerprint         CHAR(64) NOT NULL,
    device_id                       VARCHAR(255) NOT NULL,
    status                          VARCHAR(32) NOT NULL,
    transaction_data                JSONB NOT NULL,
    metadata                        JSONB NOT NULL
);

CREATE INDEX ix_pix_transaction_source_date_status
    ON pix_transaction(source_document_number, transaction_date, status);