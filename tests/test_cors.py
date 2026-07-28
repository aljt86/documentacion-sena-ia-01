import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient

from api.main import app


def test_register_allows_localhost_frontend_origin():
    client = TestClient(app)

    response = client.options(
        "/register",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_register_accepts_frontend_style_payload():
    client = TestClient(app)

    response = client.post(
        "/register",
        json={
            "name": "Ana",
            "email": "ana@frontend.com",
            "password": "123456",
            "lastName": "García",
        },
    )

    assert response.status_code == 200
    assert response.json()["mensaje"] == "Usuario registrado correctamente"
