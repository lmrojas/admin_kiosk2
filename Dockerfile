# Dockerfile para Admin Kiosk
# Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

# Etapa de construcción
FROM python:3.9-slim as builder

WORKDIR /app

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Etapa final
FROM python:3.9-slim

WORKDIR /app

# Copiar wheels de la etapa de construcción
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Instalar dependencias
RUN pip install --no-cache /wheels/*

# Copiar código fuente
COPY . .

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Exponer puerto
EXPOSE ${PORT}

# Script de inicio
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"] 