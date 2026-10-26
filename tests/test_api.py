import pytest
from fastapi.testclient import TestClient
from src.api.app import app, model_storage

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    # Inicializamos el ciclo de vida de la aplicacion
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint():
    # Comprobamos que el endpoint /health responda 200 con status ok
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_metrics_endpoint():
    # Comprobamos que el endpoint /metrics de Prometheus responda 200
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_prediction_requests_total" in response.text

def test_model_info_endpoint():
    # Comprobamos que retorne los metadatos del modelo champion
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert data["alias"] == "champion"

def test_predict_success():
    # Caso 1: Peticion valida -> 200 OK
    sample_payload = {
        "age": 0.038076,
        "sex": 0.050680,
        "bmi": 0.061696,
        "bp": 0.021872,
        "s1": -0.044223,
        "s2": -0.034821,
        "s3": -0.043401,
        "s4": -0.002592,
        "s5": 0.019907,
        "s6": -0.017646
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "model_version" in data
    assert isinstance(data["prediction"], (int, float))

def test_predict_validation_error():
    # Caso 2: Peticion invalida (falta 'bmi') -> 422 Unprocessable Entity
    invalid_payload = {
        "age": 0.038076,
        "sex": 0.050680,
        # 'bmi' ausente
        "bp": 0.021872,
        "s1": -0.044223,
        "s2": -0.034821,
        "s3": -0.043401,
        "s4": -0.002592,
        "s5": 0.019907,
        "s6": -0.017646
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_invalid_data_type():
    # Caso 3: Peticion con tipo de dato no convertible -> 422
    invalid_payload = {
        "age": "invalid_string_age",
        "sex": 0.050680,
        "bmi": 0.061696,
        "bp": 0.021872,
        "s1": -0.044223,
        "s2": -0.034821,
        "s3": -0.043401,
        "s4": -0.002592,
        "s5": 0.019907,
        "s6": -0.017646
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_model_not_available():
    # Caso 4: Modelo no disponible en memoria -> 503 Service Unavailable
    original_model = model_storage["model"]
    try:
        model_storage["model"] = None
        sample_payload = {
            "age": 0.038076, "sex": 0.050680, "bmi": 0.061696, "bp": 0.021872,
            "s1": -0.044223, "s2": -0.034821, "s3": -0.043401, "s4": -0.002592,
            "s5": 0.019907, "s6": -0.017646
        }
        response = client.post("/predict", json=sample_payload)
        assert response.status_code == 503
        
        info_resp = client.get("/model-info")
        assert info_resp.status_code == 503
    finally:
        model_storage["model"] = original_model
