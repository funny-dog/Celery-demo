from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_endpoint_exists_for_embedded_frontend():
    response = client.get("/")
    assert response.status_code == 200
