import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def train_and_track_models(
    data_path: str = "data/processed/diabetes_cleaned.csv",
    experiment_name: str = "diabetes-progression-experiment"
):
    # Configuramos el experimento en MLflow
    mlflow.set_experiment(experiment_name)
    
    # Cargamos el dataset preprocesado
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Dividimos los datos en entrenamiento y prueba con random state fijo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Definimos los modelos a entrenar y comparar
    model_configs = [
        {
            "name": "LinearRegression",
            "model": LinearRegression(),
            "params": {"fit_intercept": True}
        },
        {
            "name": "RidgeRegression",
            "model": Ridge(alpha=1.0, random_state=42),
            "params": {"alpha": 1.0, "random_state": 42}
        },
        {
            "name": "RandomForestRegressor",
            "model": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42}
        }
    ]
    
    # Entrenamos y registramos cada modelo con MLflow
    for cfg in model_configs:
        model_name = cfg["name"]
        model = cfg["model"]
        params = cfg["params"]
        
        with mlflow.start_run(run_name=model_name) as run:
            # Ajustamos el modelo con los datos de entrenamiento
            model.fit(X_train, y_train)
            
            # Generamos predicciones en el conjunto de prueba
            preds = model.predict(X_test)
            
            # Calculamos metricas de evaluacion
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            
            # Registramos parametros y metricas en MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Inferimos la firma de entrada y salida
            signature = infer_signature(X_train, preds)
            
            # Guardamos el modelo como artefacto de MLflow usando pickle
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train.iloc[:2],
                serialization_format="pickle"
            )
            
            print(f"[{model_name}] Run ID: {run.info.run_id} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")

if __name__ == "__main__":
    train_and_track_models()
