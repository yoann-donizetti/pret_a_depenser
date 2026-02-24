# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

@pytest.fixture()
def client():
    # disable lifespan to avoid loading model + init_db during unit tests
    app = create_app(enable_lifespan=False)
    return TestClient(app)