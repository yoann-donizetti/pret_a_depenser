CREATE TABLE IF NOT EXISTS prod_requests (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    endpoint TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms DOUBLE PRECISION,
    sk_id_curr TEXT,
    inputs JSONB,
    outputs JSONB,
    error TEXT,
    message TEXT
);



-- Indexes
CREATE INDEX IF NOT EXISTS idx_prod_requests_ts ON prod_requests(ts);
CREATE INDEX IF NOT EXISTS idx_prod_requests_endpoint ON prod_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_prod_requests_status_code ON prod_requests(status_code);