
"""
Tests d'intégration pour l'API principale (endpoints /health et /predict).
Vérifie les différents scénarios de succès, d'erreur et de robustesse de l'API.
"""
from fastapi.testclient import TestClient
import app.main as mainmod

class DummyModel:
    """
    Classe factice pour simuler un modèle de prédiction dans les tests.
    """
    def predict_proba(self, X):
        return [[0.2, 0.8]]

def make_client():
    """
    Crée un client de test FastAPI sans gestion du cycle de vie (lifespan désactivé).
    """
    return TestClient(mainmod.create_app(enable_lifespan=False))

def set_ready():
    """
    Met l'API dans un état 'prêt' en initialisant les variables globales nécessaires.
    """
    mainmod.MODEL = DummyModel()
    mainmod.KEPT_FEATURES = ["A"]
    mainmod.CAT_FEATURES = []
    mainmod.THRESHOLD = 0.5

def test_health_not_ready(monkeypatch):
    """
    Vérifie que l'endpoint /health retourne 'not_ready' si le modèle ou les artefacts ne sont pas chargés.
    """
    mainmod.MODEL = None
    mainmod.KEPT_FEATURES = None
    mainmod.CAT_FEATURES = None
    mainmod.THRESHOLD = None
    client = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "not_ready"

def test_predict_not_ready_503(monkeypatch):
    """
    Vérifie que l'endpoint /predict retourne un code 503 si l'API n'est pas prête.
    """
    mainmod.MODEL = None
    mainmod.KEPT_FEATURES = None
    mainmod.CAT_FEATURES = None
    mainmod.THRESHOLD = None
    client = make_client()
    r = client.post("/predict", json={"SK_ID_CURR": 1})
    assert r.status_code == 503

def test_predict_success(monkeypatch):
    """
    Vérifie qu'une prédiction aboutit avec succès et retourne une décision attendue.
    """
    set_ready()

    monkeypatch.setattr(mainmod, "get_features_by_id", lambda sk: {"SK_ID_CURR": sk, "A": 1})
    monkeypatch.setattr(mainmod, "insert_prod_request", lambda event: None)

    client = make_client()
    r = client.post("/predict", json={"SK_ID_CURR": 100001})
    assert r.status_code == 200
    assert r.json()["decision"] in ("ACCEPTED", "REFUSED")

def test_predict_404_if_id_not_found(monkeypatch):
    """
    Vérifie que l'API retourne un code 404 si l'identifiant client n'est pas trouvé en base.
    """
    set_ready()

    monkeypatch.setattr(mainmod, "get_features_by_id", lambda sk: None)
    monkeypatch.setattr(mainmod, "insert_prod_request", lambda event: None)

    client = make_client()
    r = client.post("/predict", json={"SK_ID_CURR": 999})
    assert r.status_code == 404

def test_predict_apierror_path(monkeypatch):
    """
    Vérifie que l'API retourne une erreur 400 ou 422 si les features sont incomplètes (ex : champ manquant).
    """
    set_ready()

    # manque "A" => validate_payload lève ApiError(MISSING_FIELDS)
    monkeypatch.setattr(mainmod, "get_features_by_id", lambda sk: {"SK_ID_CURR": sk})
    monkeypatch.setattr(mainmod, "insert_prod_request", lambda event: None)

    client = make_client()
    r = client.post("/predict", json={"SK_ID_CURR": 1})
    assert r.status_code in (400, 422)

def test_predict_internal_error(monkeypatch):
    """
    Vérifie que l'API retourne un code 500 en cas d'erreur interne inattendue lors de la prédiction.
    """
    set_ready()

    def boom(_sk):
        raise RuntimeError("boom")

    monkeypatch.setattr(mainmod, "get_features_by_id", boom)
    monkeypatch.setattr(mainmod, "insert_prod_request", lambda event: None)

    client = make_client()
    r = client.post("/predict", json={"SK_ID_CURR": 1})
    assert r.status_code == 500

def test_safe_log_does_not_crash(monkeypatch):
    """
    Vérifie que l'API ne plante pas si insert_prod_request lève une exception lors du logging.
    """
    set_ready()

    monkeypatch.setattr(mainmod, "get_features_by_id", lambda sk: {"SK_ID_CURR": sk, "A": 1})

    def log_boom(_event):
        raise RuntimeError("log down")

    monkeypatch.setattr(mainmod, "insert_prod_request", log_boom)

    client = make_client()
    r = client.post("/predict", json={"SK_ID_CURR": 1})
    assert r.status_code == 200