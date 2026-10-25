from pathlib import Path
import pandas as pd
from sklearn.datasets import load_diabetes

def extract_data(output_path: str = "data/raw/diabetes_raw.csv") -> pd.DataFrame:
    # Cargamos el dataset diabetes desde scikit-learn
    diabetes = load_diabetes()
    
    # Construimos el dataframe con las variables predictoras
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    
    # Anadimos la columna objetivo (progresion de la diabetes)
    df["target"] = diabetes.target
    
    # Creamos el directorio de salida si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Guardamos los datos en formato CSV en la ruta de datos raw
    df.to_csv(output_path, index=False)
    print(f"Extraccion completada con exito. Datos guardados en: {output_path}")
    print(f"Dimensiones del dataset: {df.shape}")
    return df

if __name__ == "__main__":
    extract_data()
