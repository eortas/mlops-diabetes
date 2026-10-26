FROM python:3.11-slim

WORKDIR /app

# Instalamos curl para healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiamos archivo de requerimientos e instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el codigo fuente y carpetas necesarias
COPY src/ /app/src/
COPY data/ /app/data/
COPY mlruns/ /app/mlruns/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
