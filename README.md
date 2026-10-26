# mlops-diabetes

Solucion completa de MLOps para un problema de regresion usando el dataset de diabetes de scikit-learn. El flujo cubre desde la exploracion inicial en Jupyter Notebook, validacion y limpieza de datos, entrenamiento con seguimiento de experimentos en MLflow, registro y seleccion de modelo campeon mediante Quality Gate, despliegue con FastAPI, observabilidad con Prometheus y Grafana, pruebas automatizadas con Pytest y pipeline de CI/CD con GitHub Actions.

---

## 1. Objetivo del Proyecto

El objetivo es predecir la medida cuantitativa de progresión de la diabetes un año después del registro inicial, a partir de variables clínicas y demográficas de cada paciente (edad, sexo, índice de masa corporal, presión arterial y seis mediciones de suero sanguíneo).

---

## 2. Dataset Utilizado

- **Nombre:** Diabetes Dataset (scikit-learn).
- **Muestras:** 442 pacientes.
- **Variables Predictoras (10):** `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6` (valores numéricos estandarizados).
- **Variable Objetivo (`target`):** Medida cuantitativa continua de la progresión de la enfermedad.

---

## 3. Estructura del Repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml                     # Pipeline CI/CD con GitHub Actions
├── data/
│   ├── raw/
│   │   └── diabetes_raw.csv           # Datos crudos extraídos
│   └── processed/
│       └── diabetes_cleaned.csv       # Datos limpios y listos para entrenar
├── monitoring/
│   ├── prometheus.yml                 # Configuración de recolección de Prometheus
│   └── grafana_datasources.yml        # Conexión automática de Prometheus en Grafana
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_extraction.py         # Extracción y guardado en data/raw/
│   │   ├── data_validation.py         # Comprobación de esquema y tipos
│   │   └── data_cleaning.py           # Limpieza y guardado en data/processed/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                   # Entrenamiento y tracking con MLflow
│   │   └── evaluate_and_register.py   # Quality Gate y asignación de 'champion'
│   └── api/
│       ├── __init__.py
│       ├── schemas.py                 # Validación Pydantic v2
│       └── app.py                     # API FastAPI y métricas Prometheus
├── tests/
│   └── test_api.py                    # Pruebas unitarias y de integración de la API
├── Dockerfile                         # Contenedor para desplegar la API
├── docker-compose.yml                 # Orquestación de FastAPI, Prometheus y Grafana
├── requirements.txt                   # Dependencias del proyecto
├── run_pipeline.py                    # Script orquestador del pipeline completo
├── Proyecto_MLOps.ipynb               # Notebook de exploración inicial
└── README.md                          # Documentación del proyecto
```

---

## 4. Instalación de Dependencias

Se recomienda crear un entorno virtual con Python 3.11:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual en Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Activar entorno virtual en Linux/Mac:
source venv/bin/activate

# Instalar dependencias requeridas
pip install -r requirements.txt
```

---

## 5. Ejecución del Pipeline de Datos y Modelado

Podemos ejecutar todo el flujo de forma automática o paso a paso:

### Opción A: Orquestación completa

```bash
python run_pipeline.py
```

### Opción B: Ejecución paso a paso

1. **Extracción de datos:**
   ```bash
   python -m src.data.data_extraction
   ```
2. **Validación de datos:**
   ```bash
   python -m src.data.data_validation
   ```
3. **Limpieza de datos:**
   ```bash
   python -m src.data.data_cleaning
   ```
4. **Entrenamiento y Tracking con MLflow:**
   ```bash
   python -m src.models.train
   ```
5. **Evaluación de Quality Gate y Registro de Champion:**
   ```bash
   python -m src.models.evaluate_and_register
   ```

Para inspeccionar las corridas, métricas y modelos en la interfaz visual de MLflow:

```bash
mlflow ui
```
Acceder en el navegador a: `http://localhost:5000`

---

## 6. Criterio de Selección del Modelo Campeón (Quality Gate)

El script `src/models/evaluate_and_register.py` busca la corrida con menor **RMSE** en el experimento `diabetes-progression-experiment`.

- **Quality Gate:** El modelo seleccionado debe cumplir con un umbral de `RMSE <= 58.0`.
- **Justificación del umbral:** La desviación estándar de la variable objetivo es ~77. Un RMSE inferior a 58 garantiza un error de predicción considerablemente inferior a la variabilidad base del dataset.
- Si el modelo cumple el criterio, se registra en el **MLflow Model Registry** con el nombre `diabetes-champion-model` y se le asigna el alias `@champion`.

---

## 7. Despliegue y Ejecución de la API (FastAPI)

Para iniciar el servidor localmente:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva disponible en: `http://localhost:8000/docs`

---

## 8. Ejemplos de Peticiones a los Endpoints

### 1. `GET /health`
Comprueba si la API se encuentra operativa.

```bash
curl -X GET http://localhost:8000/health
```

Respuesta esperada (`200 OK`):
```json
{
  "status": "ok"
}
```

### 2. `GET /model-info`
Devuelve el nombre del modelo, versión activa, alias y métricas registradas en MLflow.

```bash
curl -X GET http://localhost:8000/model-info
```

Respuesta esperada (`200 OK`):
```json
{
  "model_name": "diabetes-champion-model",
  "model_version": "1",
  "alias": "champion",
  "metrics": {
    "rmse": 53.68,
    "mae": 43.56,
    "r2": 0.4561
  },
  "run_id": "fc6fc8289e9a42baa574facd8be4ee37"
}
```

### 3. `POST /predict`
Recibe los valores clínicos del paciente y devuelve la predicción.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Respuesta esperada (`200 OK`):
```json
{
  "prediction": 154.23,
  "model_version": "1",
  "model_name": "diabetes-champion-model"
}
```

### 4. `GET /metrics`
Endpoint con métricas en formato Prometheus (conteo de peticiones y latencia).

```bash
curl -X GET http://localhost:8000/metrics
```

---

## 9. Manejo de Errores en la API

La API maneja de forma controlada los principales escenarios de fallo:

| Escenario | Código HTTP | Descripción |
| :--- | :--- | :--- |
| **Petición válida** | `200 OK` | Datos correctos, retorna la predicción. |
| **Campos faltantes o tipos incorrectos** | `422 Unprocessable Entity` | Pydantic valida los tipos de cada variable predictora. |
| **Modelo no disponible** | `503 Service Unavailable` | El modelo champion no se encuentra en memoria o falló su carga. |
| **Fallo en inferencia** | `500 Internal Server Error` | Ocurrió un error inesperado al calcular la predicción. |

---

## 10. Monitorización con Prometheus y Grafana

El proyecto incluye la infraestructura para monitorizar la API en tiempo real con Docker Compose:

```bash
docker-compose up --build -d
```

- **FastAPI API:** `http://localhost:8000`
- **Prometheus:** `http://localhost:9090` (Job `fastapi-mlops` scraping `/metrics`)
- **Grafana:** `http://localhost:3000` (Usuario: `admin`, Contraseña: `admin`, Prometheus ya conectado por defecto)

Métricas clave exportadas:
- `model_prediction_requests_total`: Número de llamadas a predicción según resultado (`success`, `error_500`, `error_503`).
- `model_prediction_latency_seconds`: Histograma de tiempo de respuesta de las inferencias.

---

## 11. Ejecución de Pruebas Automatizadas

Ejecución de la suite de pruebas unitarias y de integración:

```bash
python -m pytest tests/ -v
```

---

## 12. Automatización CI/CD con GitHub Actions

El archivo [`.github/workflows/ci.yml`](.github/workflows/ci.yml) automatiza el ciclo de integración continua ante cada evento `push` o `pull_request` a las ramas `main` o `master`:

1. Descarga el código (`checkout`).
2. Configura Python 3.11 y la caché de `pip`.
3. Instala las dependencias de `requirements.txt`.
4. Ejecuta secuencialmente la extracción, validación y limpieza de datos.
5. Entrena los modelos con MLflow.
6. Evalúa el Quality Gate y registra el modelo champion.
7. Ejecuta la suite de pruebas automatizadas con `pytest`.
