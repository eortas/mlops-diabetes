from src.data.data_extraction import extract_data
from src.data.data_validation import validate_data
from src.data.data_cleaning import clean_data
from src.models.train import train_and_track_models
from src.models.evaluate_and_register import evaluate_and_register_champion

def run_pipeline():
    print("=== PASO 1: Extraccion de datos ===")
    extract_data()
    
    print("\n=== PASO 2: Validacion de datos ===")
    validate_data()
    
    print("\n=== PASO 3: Limpieza de datos ===")
    clean_data()
    
    print("\n=== PASO 4: Entrenamiento y tracking de modelos con MLflow ===")
    train_and_track_models()
    
    print("\n=== PASO 5: Evaluacion contra Quality Gate y registro de Champion ===")
    champion_info = evaluate_and_register_champion()
    
    print("\nPipeline completado con exito.")
    print(f"Modelo Champion activo: {champion_info['model_name']} version {champion_info['version']}")

if __name__ == "__main__":
    run_pipeline()
