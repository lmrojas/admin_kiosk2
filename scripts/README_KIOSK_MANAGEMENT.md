# Sistema de Gestión Integral de Kiosks

## Descripción General

El script `kiosk_management.py` proporciona un sistema integral para monitorear, gestionar y recuperar kiosks en el sistema Admin Kiosk. Combina funcionalidades de monitoreo y recuperación de desastres en una herramienta unificada.

## Características Principales

- Chequeo de salud de kiosks
- Detección de kiosks inactivos
- Respaldo automático de kiosks
- Estrategias de recuperación flexibles
- Generación de informes detallados

## Requisitos Previos

- Python 3.9+
- Flask
- Módulos de monitoreo y recuperación de desastres

## Modos de Operación

### 1. Chequeo de Salud

```bash
python scripts/kiosk_management.py --action health-check [--inactivity-days 7]
```

- Genera un informe de estado de todos los kiosks
- Detecta kiosks inactivos
- Opcional: Respalda kiosks inactivos

### 2. Recuperación de Kiosks

```bash
python scripts/kiosk_management.py --action recover [--inactivity-days 7] [--recovery-strategy backup|reset|notify]
```

Estrategias de recuperación:
- `backup`: Crear respaldo de kiosks inactivos
- `reset`: Reiniciar kiosks inactivos (pendiente de implementación)
- `notify`: Enviar alertas sobre kiosks inactivos

### 3. Gestión Completa

```bash
python scripts/kiosk_management.py --action full-management [--inactivity-days 7] [--recovery-strategy backup|reset|notify]
```

- Realiza chequeo de salud
- Aplica estrategia de recuperación
- Genera informe completo

## Configuración

### Parámetros

- `--action`: Modo de operación (`health-check`, `recover`, `full-management`)
- `--inactivity-days`: Días de inactividad para considerar un kiosk inactivo (default: 7)
- `--recovery-strategy`: Estrategia de recuperación (default: `backup`)

## Salida

La herramienta genera informes en formato JSON con:
- Marca de tiempo
- Estado de salud de kiosks
- Kiosks inactivos
- Resultados de recuperación

## Consideraciones de Seguridad

- Mantén los scripts actualizados
- Protege los directorios de respaldo
- Configura alertas y notificaciones

## Registro de Eventos

Los eventos se registran utilizando el módulo `logging`. Configura los niveles de registro según tus necesidades.

## Solución de Problemas

- Verifica permisos de archivos y directorios
- Comprueba la conectividad de los kiosks
- Revisa los registros de eventos

## Contribución

Si encuentras errores o tienes sugerencias de mejora, por favor abre un issue o envía un pull request.

## Licencia

[Especificar la licencia del proyecto]

## Descargo de Responsabilidad

Este script se proporciona "tal cual" sin garantías. Siempre realiza pruebas exhaustivas en un entorno de staging antes de usar en producción. 