# Integración Continua y Despliegue Continuo (CI/CD)

## Descripción General

Este proyecto utiliza GitHub Actions para implementar un flujo de trabajo de Integración Continua y Despliegue Continuo (CI/CD) para la aplicación Admin Kiosk.

## Flujo de Trabajo

### Ramas
- `develop`: Rama de desarrollo continuo
- `main`: Rama de producción estable

### Procesos de CI/CD

#### 1. Pruebas
- Ejecución de pruebas unitarias e integración
- Soporte para Python 3.9, 3.10, y 3.11
- Generación de informe de cobertura de código

#### 2. Verificaciones de Seguridad
- Análisis de seguridad con Bandit
- Verificación de dependencias con Safety

#### 3. Linting
- Verificación de formato con Black
- Análisis de código con Flake8

#### 4. Despliegue
- Despliegue automático en entorno de staging desde la rama `develop`
- Despliegue automático en producción desde la rama `main`

## Secretos Requeridos

Para configurar el workflow, se necesitan los siguientes secretos en GitHub:

- `STAGING_DEPLOY_KEY`: Clave de despliegue para entorno de staging
- `STAGING_HOST`: Host del servidor de staging
- `STAGING_USER`: Usuario de despliegue en staging
- `PRODUCTION_DEPLOY_KEY`: Clave de despliegue para producción
- `PRODUCTION_HOST`: Host del servidor de producción
- `PRODUCTION_USER`: Usuario de despliegue en producción
- `SLACK_WEBHOOK`: URL del webhook de Slack para notificaciones

## Ejecución Manual

Puedes ejecutar los scripts de CI/CD manualmente:

```bash
# Ejecutar pruebas
pytest tests/unit tests/integration

# Verificación de seguridad
bandit -r app
safety check

# Linting
black --check .
flake8 app tests

# Despliegue
python scripts/deploy.py deploy --env staging
python scripts/deploy.py deploy --env production
```

## Configuración Adicional

- Asegúrate de tener instaladas todas las dependencias de desarrollo
- Configura los archivos `.env` correspondientes para cada entorno
- Mantén actualizadas las dependencias con `pip install -r requirements.txt`

## Solución de Problemas

- Revisa los logs de GitHub Actions en caso de fallos
- Verifica la configuración de secretos
- Comprueba la compatibilidad de versiones de Python

## Contribución

Para contribuir al proceso de CI/CD:
1. Crea una rama desde `develop`
2. Implementa tus cambios
3. Asegúrate de que todas las pruebas pasen
4. Crea un Pull Request a `develop` 