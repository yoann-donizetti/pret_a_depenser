# tests/test_health_endpoint.py
def test_health_returns_ok_or_not_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "not_ready")