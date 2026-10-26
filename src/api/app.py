import time
from typing import Dict, Any
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.api.schemas import DiabetesFeatures, PredictionResponse, ModelInfoResponse
from src.models.evaluate_and_register import MODEL_REGISTRY_NAME

# Definimos metricas de Prometheus para monitorizacion
PREDICTION_REQUESTS = Counter(
    "model_prediction_requests_total",
    "Total de peticiones al endpoint de prediccion",
    ["status"]
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Tiempo de respuesta para generar predicciones"
)

# Estructura global para almacenar el modelo y sus metadatos
model_storage: Dict[str, Any] = {
    "model": None,
    "model_name": MODEL_REGISTRY_NAME,
    "version": None,
    "alias": "champion",
    "metrics": {},
    "run_id": None
}

def load_champion_model():
    # Cargamos el modelo marcado con el alias champion desde MLflow
    try:
        client = MlflowClient()
        # Obtenemos la version asociada al alias champion
        model_version_details = client.get_model_version_by_alias(
            name=MODEL_REGISTRY_NAME,
            alias="champion"
        )
        version = str(model_version_details.version)
        run_id = model_version_details.run_id
        
        # Obtenemos metricas de la corrida en MLflow
        run_data = client.get_run(run_id).data
        metrics = run_data.metrics
        
        # Cargamos el modelo entrenado
        model_uri = f"models:/{MODEL_REGISTRY_NAME}@champion"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        model_storage["model"] = loaded_model
        model_storage["version"] = version
        model_storage["run_id"] = run_id
        model_storage["metrics"] = metrics
        print(f"Modelo champion version {version} cargado correctamente.")
    except Exception as e:
        print(f"Aviso: No se pudo cargar el modelo champion al iniciar: {e}")
        model_storage["model"] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga de recursos al arrancar la aplicacion
    load_champion_model()
    yield

# Creamos la instancia de FastAPI con metadatos descriptivos
app = FastAPI(
    title="Diabetes MLOps Prediction API",
    description="API REST para servir el modelo campeon de prediccion de progresion de diabetes con MLflow y Prometheus",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", summary="Comprobar el estado del servicio")
def health():
    # Endpoint basico para verificar disponibilidad
    return {"status": "ok"}

@app.get("/model-info", response_model=ModelInfoResponse, summary="Obtener informacion del modelo activo")
def model_info():
    # Validamos si el modelo esta disponible en memoria
    if model_storage["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo champion no esta cargado o disponible en el registro."
        )
    return {
        "model_name": model_storage["model_name"],
        "model_version": model_storage["version"],
        "alias": model_storage["alias"],
        "metrics": model_storage["metrics"],
        "run_id": model_storage["run_id"]
    }

@app.post("/predict", response_model=PredictionResponse, summary="Generar prediccion de progresion")
def predict(features: DiabetesFeatures):
    # 1. Comprobamos si el servicio tiene el modelo cargado
    if model_storage["model"] is None:
        PREDICTION_REQUESTS.labels(status="error_503").inc()
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: el modelo champion no ha sido cargado."
        )
    
    start_time = time.time()
    try:
        # 2. Convertimos las caracteristicas recibidas a DataFrame de pandas
        input_data = pd.DataFrame([features.model_dump()])
        
        # 3. Realizamos la inferencia con el modelo cargado
        prediction_val = model_storage["model"].predict(input_data)
        prediction_float = float(prediction_val[0])
        
        # Registramos las metricas de observabilidad
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_REQUESTS.labels(status="success").inc()
        
        return {
            "prediction": round(prediction_float, 2),
            "model_version": str(model_storage["version"]),
            "model_name": str(model_storage["model_name"])
        }
    except Exception as e:
        PREDICTION_REQUESTS.labels(status="error_500").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Fallo durante la generacion de la prediccion: {str(e)}"
        )

@app.get("/metrics", summary="Endpoint de metricas para Prometheus")
def metrics():
    # Exponemos las metricas en formato compatible con Prometheus
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
