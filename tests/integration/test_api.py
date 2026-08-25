"""Integration tests for the bounded, research-only REST API."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from myoadapt.api.rest import MAX_BATCH_SAMPLES, create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(scope="module")
def protected_client() -> TestClient:
    return TestClient(create_app(api_key="test-api-key"))


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["version"] == "2.0.0"
    assert body["research_use_only"] is True


def test_root_endpoint_discloses_research_boundary(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Research" in resp.text
    assert "clinical" in resp.text.lower()


def test_list_models(client: TestClient) -> None:
    resp = client.get("/models")
    assert resp.status_code == 200
    assert "classical" in resp.json()["models"]


def test_list_datasets(client: TestClient) -> None:
    resp = client.get("/datasets")
    body = resp.json()
    assert resp.status_code == 200
    assert "DB2" in body["datasets"]
    assert body["metadata"]["DB2"]["n_channels"] == 12


def test_explicit_api_key_is_enforced(protected_client: TestClient) -> None:
    assert protected_client.get("/models").status_code == 401
    assert protected_client.get("/models", headers={"X-API-Key": "wrong"}).status_code == 401
    assert protected_client.get("/models", headers={"X-API-Key": "test-api-key"}).status_code == 200


def test_train_endpoint_is_explicitly_disabled(client: TestClient) -> None:
    response = client.post("/train", json={"dataset": "DB2", "model_type": "xgboost"})
    assert response.status_code == 501
    assert "disabled" in response.json()["detail"].lower()


def test_predict_rejects_non_finite_input_before_model_call(client: TestClient) -> None:
    response = client.post(
        "/predict",
        content='{"features": [1.0, NaN]}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_batch_rejects_ragged_input_before_model_call(client: TestClient) -> None:
    response = client.post("/predict_batch", json={"features": [[1.0, 2.0], [3.0]]})
    assert response.status_code == 422


def test_predict_batch_rejects_oversized_batch_before_model_call(client: TestClient) -> None:
    response = client.post("/predict_batch", json={"features": [[1.0]] * (MAX_BATCH_SAMPLES + 1)})
    assert response.status_code == 422


def test_predict_requires_a_loaded_model_after_validation(client: TestClient) -> None:
    response = client.post("/predict", json={"features": [0.1, 0.2]})
    assert response.status_code == 503


def test_openapi_schema(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "MyoAdapt Research API"
    assert {"/health", "/models", "/datasets", "/train", "/predict"}.issubset(schema["paths"])
