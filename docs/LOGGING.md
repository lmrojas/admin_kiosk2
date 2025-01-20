# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

# Sistema de Logging Centralizado

## Descripción General
El sistema de logging centralizado proporciona una manera estructurada y consistente de registrar eventos, errores y actividades de auditoría en la aplicación.

## Tipos de Logs

### 1. Logs de Seguridad (`security.log`)
- Eventos de autenticación
- Intentos de acceso
- Cambios en permisos/roles
- Rate limiting

### 2. Logs de Auditoría (`audit.log`)
- Acciones de usuarios
- Cambios en recursos
- Operaciones CRUD
- Trazabilidad de cambios

### 3. Logs de Error (`error.log`)
- Excepciones
- Errores de sistema
- Fallos de integración
- Stack traces

## Uso del Servicio de Logging

### Importación
```python
from app.services.logging_service import LoggingService
```

### Eventos de Seguridad
```python
logging_service = LoggingService()
logging_service.log_security_event(
    event_type='login_attempt',
    details={'username': 'user@example.com', 'success': True}
)
```

### Eventos de Auditoría
```python
logging_service.log_audit_event(
    user_id=1,
    action='create_kiosk',
    resource='kiosk',
    status='success',
    details={'kiosk_id': 123}
)
```

### Errores
```python
try:
    # código que puede fallar
except Exception as e:
    logging_service.log_error(
        error=e,
        context={'function': 'process_kiosk_data'}
    )
```

## Formato de Logs
Los logs se almacenan en formato JSON con la siguiente estructura:

### Seguridad
```json
{
    "timestamp": "2024-01-17T10:30:00",
    "event_type": "login_attempt",
    "ip": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "details": {
        "username": "user@example.com",
        "success": true
    }
}
```

### Auditoría
```json
{
    "timestamp": "2024-01-17T10:30:00",
    "user_id": 1,
    "action": "create_kiosk",
    "resource": "kiosk",
    "status": "success",
    "ip": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "details": {
        "kiosk_id": 123
    }
}
```

### Error
```json
{
    "timestamp": "2024-01-17T10:30:00",
    "error_type": "ValueError",
    "error_message": "Invalid kiosk data",
    "ip": "192.168.1.1",
    "endpoint": "/api/kiosk",
    "method": "POST",
    "context": {
        "function": "process_kiosk_data"
    }
}
```

## Rotación de Logs
- Los archivos de log rotan automáticamente al alcanzar 10MB
- Se mantienen hasta 10 archivos de backup
- Formato de archivo rotado: `{nombre}.log.1`, `{nombre}.log.2`, etc.

## Consulta de Logs
```python
# Obtener logs filtrados por fecha
logs = logging_service.get_logs(
    log_type='security',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    limit=100
)
```

## Mejores Prácticas
1. **Seguridad**:
   - No registrar contraseñas ni datos sensibles
   - Usar niveles apropiados de logging
   - Mantener separación entre tipos de logs

2. **Rendimiento**:
   - Los logs son asíncronos para no bloquear
   - Usar rate limiting en producción
   - Monitorear el espacio en disco

3. **Mantenimiento**:
   - Revisar logs regularmente
   - Configurar alertas para errores críticos
   - Mantener backups de logs importantes
``` 