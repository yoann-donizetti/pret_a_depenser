-- 002_init_features_store.sql

CREATE TABLE IF NOT EXISTS features_store (
  sk_id_curr BIGINT PRIMARY KEY,
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_features_store_data_gin
ON features_store USING GIN (data);

DO $$
BEGIN
  -- fonction dédiée
  CREATE OR REPLACE FUNCTION features_store_set_updated_at()
  RETURNS TRIGGER AS $f$
  BEGIN
    NEW.updated_at = now();
    RETURN NEW;
  END;
  $f$ LANGUAGE plpgsql;

  -- trigger (créé une seule fois)
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_features_store_updated_at'
  ) THEN
    CREATE TRIGGER trg_features_store_updated_at
    BEFORE UPDATE ON features_store
    FOR EACH ROW
    EXECUTE FUNCTION features_store_set_updated_at();
  END IF;
END $$;