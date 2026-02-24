INSERT INTO features_store (sk_id_curr, data)
VALUES (%(sk_id_curr)s, %(data)s)
ON CONFLICT (sk_id_curr) DO UPDATE SET
  data = EXCLUDED.data,
  updated_at = now();