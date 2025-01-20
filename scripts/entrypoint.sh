#!/bin/bash

# Script de punto de entrada para contenedor Docker
#
# Funcionalidad:
# - Inicializa el contenedor Docker
# - Espera a que la base de datos esté lista
# - Ejecuta migraciones pendientes
# - Inicia el servidor de aplicación
# - Maneja señales de apagado
#
# Uso:
# ./scripts/entrypoint.sh
#
# Variables de entorno:
# - FLASK_APP: Nombre de la aplicación
# - FLASK_ENV: Ambiente (development/production)
# - DATABASE_URL: URL de conexión a la base de datos
#
# Notas:
# - Script principal para Docker
# - No ejecutar directamente en host

set -e

# Esperar a que la base de datos esté lista
echo "Esperando a que la base de datos esté lista..."
python scripts/wait_for_db.py

# Aplicar migraciones
echo "Aplicando migraciones..."
python manage.py migrate --noinput

# Recolectar archivos estáticos
echo "Recolectando archivos estáticos..."
python manage.py collectstatic --noinput

# Iniciar Gunicorn
echo "Iniciando Gunicorn..."
exec gunicorn app.wsgi:application \
    --bind 0.0.0.0:${PORT} \
    --workers 4 \
    --threads 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile - 