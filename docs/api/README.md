# Documentación API - Admin Kiosk
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

## Información General

- **Base URL**: `https://api.admin-kiosk.com/v1`
- **Formato**: JSON
- **Autenticación**: JWT + 2FA
- **Rate Limiting**: 1000 requests/hora

## Autenticación

### Login
```http
POST /auth/login
```

**Request Body:**
```json
{
  "email": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "token": "string",
  "requires_2fa": boolean,
  "user": {
    "id": "string",
    "email": "string",
    "role": "string"
  }
}
```

### Verificar 2FA
```http
POST /auth/verify-2fa
```

**Request Body:**
```json
{
  "token": "string",
  "code": "string"
}
```

## Kiosks

### Listar Kiosks
```http
GET /kiosks
```

**Query Parameters:**
- `page`: número de página (default: 1)
- `limit`: resultados por página (default: 10)
- `status`: filtrar por estado
- `search`: búsqueda por texto

### Crear Kiosk
```http
POST /kiosks
```

**Request Body:**
```json
{
  "name": "string",
  "location": "string",
  "type": "string",
  "config": {
    "settings": object
  }
}
```

### Obtener Kiosk
```http
GET /kiosks/{id}
```

### Actualizar Kiosk
```http
PUT /kiosks/{id}
```

### Eliminar Kiosk
```http
DELETE /kiosks/{id}
```

## Monitoreo

### Estado del Kiosk
```http
GET /kiosks/{id}/status
```

**Response:**
```json
{
  "status": "string",
  "last_update": "datetime",
  "metrics": {
    "cpu": number,
    "memory": number,
    "disk": number
  }
}
```

### Histórico de Métricas
```http
GET /kiosks/{id}/metrics
```

**Query Parameters:**
- `start_date`: fecha inicio (ISO 8601)
- `end_date`: fecha fin (ISO 8601)
- `metric_type`: tipo de métrica

## Sistema de IA

### Predicciones
```http
POST /ai/predict
```

**Request Body:**
```json
{
  "kiosk_id": "string",
  "data": object
}
```

### Explicabilidad
```http
GET /ai/explain/{prediction_id}
```

### Métricas de Modelo
```http
GET /ai/metrics
```

## WebSocket

### Conexión en Tiempo Real
```websocket
WS /ws/kiosks/{id}
```

**Eventos:**
- `status_update`: Actualización de estado
- `alert`: Alerta nueva
- `metric_update`: Actualización de métricas

## Errores

### Códigos de Estado
- `200`: Éxito
- `201`: Creado
- `400`: Error de solicitud
- `401`: No autorizado
- `403`: Prohibido
- `404`: No encontrado
- `429`: Demasiadas solicitudes
- `500`: Error interno del servidor

### Formato de Error
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": object
  }
}
```

## Paginación

### Formato de Respuesta Paginada
```json
{
  "data": array,
  "pagination": {
    "total": number,
    "page": number,
    "limit": number,
    "pages": number
  }
}
```

## Versionado
- Versión actual: v1
- Formato de versionado: `/v{version_number}`
- Política de deprecación: 6 meses de soporte después del anuncio

## Rate Limiting
- Headers de respuesta:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

## CORS
- Dominios permitidos configurables
- Métodos permitidos: GET, POST, PUT, DELETE, OPTIONS
- Headers permitidos: Content-Type, Authorization
``` 