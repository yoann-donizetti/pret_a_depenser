SELECT
  ts,
  endpoint,
  status_code,
  latency_ms,
  sk_id_curr,
  inputs,
  outputs,
  error,
  message
FROM prod_requests
WHERE endpoint = %(endpoint)s
ORDER BY id DESC
LIMIT %(limit)s;