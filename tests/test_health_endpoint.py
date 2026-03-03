
"""
Test unitaire pour l'endpoint /health de l'API.
Vérifie que le statut retourné est bien 'ok' ou 'not_ready'.
"""
# tests/test_health_endpoint.py
def test_health_returns_ok_or_not_ready(client):
    """
    Vérifie que l'endpoint /health retourne un code 200 et un statut valide ('ok' ou 'not_ready').
    """
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "not_ready")