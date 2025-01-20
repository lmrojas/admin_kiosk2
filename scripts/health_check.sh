#!/bin/bash

# Script de verificación de salud para Admin Kiosk
# Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

# Validar argumentos
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 <environment>"
    exit 1
fi

ENVIRONMENT=$1
MAX_RETRIES=30
RETRY_INTERVAL=10

# Configurar URL según el ambiente
case $ENVIRONMENT in
    "staging")
        BASE_URL="https://staging-admin-kiosk.example.com"
        ;;
    "production")
        BASE_URL="https://admin-kiosk.example.com"
        ;;
    *)
        echo "Ambiente no válido. Use 'staging' o 'production'"
        exit 1
        ;;
esac

HEALTH_ENDPOINT="$BASE_URL/api/health"

echo "Verificando salud del servicio en $ENVIRONMENT..."
echo "URL: $HEALTH_ENDPOINT"

for i in $(seq 1 $MAX_RETRIES); do
    response=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_ENDPOINT)
    
    if [ "$response" -eq 200 ]; then
        echo "Servicio saludable después de $i intentos"
        exit 0
    fi
    
    echo "Intento $i de $MAX_RETRIES falló (código: $response)"
    
    if [ "$i" -lt "$MAX_RETRIES" ]; then
        echo "Esperando $RETRY_INTERVAL segundos antes del siguiente intento..."
        sleep $RETRY_INTERVAL
    fi
done

echo "El servicio no está saludable después de $MAX_RETRIES intentos"
exit 1 