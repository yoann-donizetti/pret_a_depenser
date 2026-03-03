"""
Fichier de configuration des tests Pytest.
Définit des fixtures réutilisables pour les tests unitaires.
"""
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

@pytest.fixture()
def client():
    """
    Fixture Pytest qui fournit un client de test FastAPI.
    Désactive le lifespan pour éviter de charger le modèle et d'initialiser la base lors des tests unitaires.
    Retourne :
        TestClient prêt à l'emploi pour les requêtes sur l'API.
    """
    app = create_app(enable_lifespan=False)
    return TestClient(app)