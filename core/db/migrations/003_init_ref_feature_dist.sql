-- 003_init_ref_feature_dist.sql

CREATE TABLE IF NOT EXISTS ref_feature_dist (

  feature TEXT PRIMARY KEY,

  kind TEXT NOT NULL CHECK (kind IN ('numeric','categorical')),

  bins_json JSONB,

  ref_dist_json JSONB NOT NULL,

  n_ref BIGINT NOT NULL,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()

);

CREATE INDEX IF NOT EXISTS idx_ref_feature_dist_kind
ON ref_feature_dist(kind);