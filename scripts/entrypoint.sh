#!/bin/bash

# Script de entrada para Admin Kiosk
# Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

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