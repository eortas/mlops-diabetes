import mlflow
from mlflow.tracking import MlflowClient

MODEL_REGISTRY_NAME = "diabetes-champion-model"
# Umbral maximo aceptable de error RMSE para pasar el quality gate
QUALITY_GATE_RMSE_THRESHOLD = 58.0

def evaluate_and_register_champion(
    experiment_name: str = "diabetes-progression-experiment",
    model_name: str = MODEL_REGISTRY_NAME,
    rmse_threshold: float = QUALITY_GATE_RMSE_THRESHOLD
):
    client = MlflowClient()
    
    # Obtenemos el experimento
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No se encontro el experimento: {experiment_name}")
        
    # Buscamos todas las ejecuciones ordenadas por menor RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"]
    )
    
    if not runs:
        raise ValueError("No se encontraron ejecuciones registradas en el experimento.")
        
    best_run = runs[0]
    best_rmse = best_run.data.metrics.get("rmse")
    best_run_id = best_run.info.run_id
    best_model_type = best_run.data.params.get("model_type", "Unknown")
    
    print(f"Mejor ejecucion encontrada: {best_run_id} ({best_model_type}) con RMSE = {best_rmse:.2f}")
    
    # Evaluamos el Quality Gate
    if best_rmse > rmse_threshold:
        raise ValueError(
            f"El mejor modelo (RMSE={best_rmse:.2f}) no supera el Quality Gate (RMSE <= {rmse_threshold}). Registro cancelado."
        )
        
    print(f"Supera el Quality Gate (RMSE <= {rmse_threshold}). Registrando en Model Registry...")
    
    # Registramos el modelo en el Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    model_version_obj = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(model_version_obj.version)
    
    # Asignamos el alias champion a la version seleccionada
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version
    )
    
    # Tambien actualizamos la descripcion de la version
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Modelo Champion ({best_model_type}) con RMSE={best_rmse:.2f}"
    )
    
    print(f"Modelo registrado exitosamente: '{model_name}' version {version} con alias 'champion'.")
    return {
        "model_name": model_name,
        "version": version,
        "alias": "champion",
        "rmse": best_rmse,
        "model_type": best_model_type,
        "run_id": best_run_id
    }

if __name__ == "__main__":
    evaluate_and_register_champion()
