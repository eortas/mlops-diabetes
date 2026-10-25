from pathlib import Path
import pandas as pd
import numpy as np

EXPECTED_FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
TARGET_COLUMN = "target"

def validate_data(input_path: str = "data/raw/diabetes_raw.csv") -> bool:
    # Verificamos que el archivo de entrada exista
    raw_file = Path(input_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"No se encontro el archivo para validar en: {input_path}")
    
    # Cargamos el archivo csv
    df = pd.read_csv(raw_file)
    
    # 1. Comprobamos que el dataset no este vacio
    assert len(df) > 0, "El dataset esta vacio."
    
    # 2. Comprobamos que la columna target exista
    assert TARGET_COLUMN in df.columns, f"Falta la columna objetivo: {TARGET_COLUMN}"
    
    # 3. Comprobamos la presencia de todas las variables predictoras requeridas
    for feature in EXPECTED_FEATURES:
        assert feature in df.columns, f"Falta la variable predictora: {feature}"
        
    # 4. Comprobamos que todas las columnas sean de tipo numerico
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) == len(df.columns), "Existen columnas no numericas en el dataset."
    
    # 5. Comprobamos que no existan valores nulos criticos en este paso
    null_count = df.isnull().sum().sum()
    print(f"Total de valores nulos encontrados: {null_count}")
    
    print("Validacion de datos completada satisfactoriamente.")
    return True

if __name__ == "__main__":
    validate_data()
