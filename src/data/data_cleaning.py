from pathlib import Path
import pandas as pd

def clean_data(
    input_path: str = "data/raw/diabetes_raw.csv",
    output_path: str = "data/processed/diabetes_cleaned.csv"
) -> pd.DataFrame:
    # Verificamos la existencia del archivo sin procesar
    raw_file = Path(input_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")
        
    df = pd.read_csv(raw_file)
    initial_rows = len(df)
    
    # 1. Eliminamos filas duplicadas si las hubiera
    df = df.drop_duplicates()
    
    # 2. Manejamos valores nulos eliminando filas incompletas
    df = df.dropna()
    
    # 3. Nos aseguramos de que los tipos sean numericos float/int
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
        
    # Guardamos los datos limpios y procesados
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Limpieza finalizada. Filas iniciales: {initial_rows}, filas finales: {len(df)}")
    print(f"Archivo limpio guardado en: {output_path}")
    return df

if __name__ == "__main__":
    clean_data()
