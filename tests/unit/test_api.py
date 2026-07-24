from __future__ import annotations

from fastapi.testclient import TestClient

from vsrx.app.api import create_app


def test_health_endpoint(fast_config) -> None:
    with TestClient(create_app(fast_config)) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["models"]["telea"] is True


def test_config_endpoint(fast_config) -> None:
    with TestClient(create_app(fast_config)) as client:
        response = client.get("/v1/config")
        assert response.status_code == 200
        payload = response.json()
        assert payload["profile"] == "fast"
        assert payload["hash"] == fast_config.hash
