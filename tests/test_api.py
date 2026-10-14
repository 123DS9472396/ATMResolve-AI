"""Tests for the FastAPI health endpoint and API-layer behaviour.

These tests use FastAPI's TestClient (Starlette) which runs requests
in-process without starting a real server.  The health probe for Ollama
is expected to report an error (or degraded) in a CI / test environment
where Ollama is not running; the test therefore checks structure rather
than asserting all components are healthy.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Return a TestClient for the FastAPI app with the real startup lifecycle."""
    from api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_health_returns_200(client):
    """Health endpoint must always return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_has_required_keys(client):
    """Health response must include top-level status, mode, and components."""
    body = client.get("/health").json()
    assert "status" in body
    assert "mode" in body
    assert "components" in body
    assert body["mode"] == "local_synthetic_demo"


def test_health_components_have_status_field(client):
    """Every component entry in the health response must have a 'status' field."""
    body = client.get("/health").json()
    for name, info in body["components"].items():
        assert "status" in info, f"Component {name!r} is missing 'status' key"


def test_health_overall_status_reflects_components(client):
    """Top-level status must be 'ok' only when all components report 'ok',
    and 'degraded' when at least one does not."""
    body = client.get("/health").json()
    component_statuses = [v["status"] for v in body["components"].values()]
    if all(s == "ok" for s in component_statuses):
        assert body["status"] == "ok"
    else:
        assert body["status"] == "degraded"


def test_health_database_component_present(client):
    """The database component must always be present and report ok when the
    application database is reachable."""
    body = client.get("/health").json()
    assert "database" in body["components"]
    # The test DB is initialised at startup; it should be reachable.
    assert body["components"]["database"]["status"] == "ok"


def test_health_ollama_component_present(client):
    """Ollama component must always be present in the health response.
    Its status may be 'error' in environments without Ollama running."""
    body = client.get("/health").json()
    assert "ollama" in body["components"]
    assert body["components"]["ollama"]["status"] in {"ok", "error"}


def test_health_chroma_component_present(client):
    """Chroma component must always be present in the health response."""
    body = client.get("/health").json()
    assert "chroma" in body["components"]
    assert body["components"]["chroma"]["status"] in {"ok", "error", "not_ingested"}
