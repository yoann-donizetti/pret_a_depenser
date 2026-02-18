import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client():
    app = create_app(enable_lifespan=False)

    # injecter globals fake ici si besoin
    import app.main as main
    main.MODEL = object()
    main.KEPT_FEATURES = ["SK_ID_CURR", "EXT_SOURCE_1"]
    main.CAT_FEATURES = []
    main.THRESHOLD = 0.5

    return TestClient(app)