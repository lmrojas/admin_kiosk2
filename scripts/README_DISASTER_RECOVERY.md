# Script de Recuperación de Desastres para Admin Kiosk

## Descripción General

El script `disaster_recovery.py` proporciona una solución integral para la gestión de respaldos y recuperación de desastres en el sistema Admin Kiosk. Permite crear respaldos completos, restaurar sistemas desde respaldos y listar respaldos disponibles.

## Características Principales

- Respaldo completo del sistema
- Restauración de respaldos
- Listado de respaldos disponibles
- Registro de eventos de recuperación
- Soporte para respaldos personalizados

## Requisitos Previos

- Python 3.9+
- Flask
- Permisos de escritura en el directorio de respaldos

## Uso

### Crear un Respaldo

```bash
python scripts/disaster_recovery.py backup [--name NOMBRE_PERSONALIZADO]
```

### Restaurar un Respaldo

```bash
python scripts/disaster_recovery.py restore --path /ruta/al/respaldo.zip
```

### Listar Respaldos Disponibles

```bash
python scripts/disaster_recovery.py list
```

## Configuración

### Directorio de Respaldos

Por defecto, los respaldos se almacenan en un directorio llamado `backups`. Puedes personalizar esto modificando el parámetro `backup_dir` en la clase `DisasterRecoveryManager`.

### Elementos Respaldados

- Directorio `app`
- Directorio `config`
- Directorio `migrations`
- Archivo `requirements.txt`
- Archivo `run.py`
- Base de datos

## Consideraciones de Seguridad

- Mantén los respaldos en una ubicación segura y protegida
- Limita el acceso al script de recuperación
- Considera cifrar respaldos que contengan información sensible

## Registro de Eventos

Los eventos de recuperación se registran utilizando el módulo `logging`. Puedes configurar el nivel de registro y el destino según tus necesidades.

## Solución de Problemas

- Verifica permisos de archivo y directorio
- Asegúrate de tener las dependencias instaladas
- Comprueba la integridad del respaldo antes de restaurar

## Contribución

Si encuentras errores o tienes sugerencias de mejora, por favor abre un issue o envía un pull request.

## Licencia

[Especificar la licencia del proyecto]

## Descargo de Responsabilidad

Este script se proporciona "tal cual" sin garantías. Siempre realiza pruebas exhaustivas en un entorno de staging antes de usar en producción. 