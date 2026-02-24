INSERT INTO ref_feature_dist(feature, kind, bins_json, ref_dist_json, n_ref)
VALUES (%(feature)s, %(kind)s, %(bins_json)s, %(ref_dist_json)s, %(n_ref)s)
ON CONFLICT (feature) DO UPDATE SET
  kind = EXCLUDED.kind,
  bins_json = EXCLUDED.bins_json,
  ref_dist_json = EXCLUDED.ref_dist_json,
  n_ref = EXCLUDED.n_ref,
  created_at = now();