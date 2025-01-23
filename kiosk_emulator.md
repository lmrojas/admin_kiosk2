# Módulo Kiosk App

Módulo para la simulación y envío de datos de kiosks. Este módulo se encarga de recolectar datos del sistema operativo y hardware, simular variaciones realistas en los datos, y enviarlos a través de WebSocket.

## Estructura

El módulo está organizado en dos componentes principales:

```
kiosk_app/
├── kiosk_app.py           # Clase principal del kiosk
├── kiosk_behavior_simulator.py  # Simulador de comportamiento
├── requirements.txt       # Dependencias del módulo
└── README.md             # Documentación
```

## Datos Enviados

El kiosk envía periódicamente la siguiente telemetría:

1. **Identificación**
   - Serial del kiosk
   - Nombre del kiosk
   - Información de red
     - IP local y pública
     - MAC address
     - Interface activa

2. **Estado del Sistema**
   - Sistema operativo (nombre y versión)
   - Uptime desde inicio
   - Estado actual del kiosk
   - Timestamp del último reinicio
   - Estado de Chromium
   - Razón de bloqueo (si aplica)
   - Hora local con zona horaria

3. **Hardware**
   - CPU
     - Porcentaje de uso
     - Temperatura
   - Memoria
     - Total disponible
     - En uso
     - Porcentaje utilizado
   - Disco
     - Espacio total
     - Espacio usado
     - Espacio libre
     - Porcentaje utilizado

4. **Red**
   - Calidad de señal WiFi
   - Velocidad de conexión
   - Latencia
   - Estado de conexión
   - Contadores de paquetes
     - Enviados
     - Recibidos
     - Perdidos

5. **Sensores**
   - Temperatura ambiente
   - Humedad
   - Voltaje
   - Estado de la puerta

6. **Seguridad**
   - Último acceso no autorizado
   - IP del intento
   - Tipo de intento
   - Contador de intentos

7. **Geolocalización**
   - Latitud
   - Longitud
   - Timestamp de última actualización

## Estados Posibles

El kiosk puede estar en los siguientes estados:
- `online`: Operativo y conectado
- `offline`: Sin conexión
- `blocked`: Bloqueado por seguridad
- `busy`: Procesando una operación
- `maintenance`: En mantenimiento
- `restarting`: En proceso de reinicio
- `updating`: Actualizando software
- `error`: Error en el sistema

## Comandos Soportados

El kiosk responde a los siguientes comandos:

1. `restart`
   - Inicia el proceso de reinicio
   - Cambia estado a "restarting"
   - Retorna confirmación con timestamp

2. `update`
   - Inicia actualización de software
   - Cambia estado a "updating"
   - Retorna confirmación con timestamp

3. `block`
   - Bloquea el kiosk
   - Requiere razón de bloqueo
   - Cambia estado a "blocked"
   - Retorna confirmación con razón

4. `unblock`
   - Desbloquea el kiosk
   - Solo funciona si está bloqueado
   - Cambia estado a "online"
   - Retorna confirmación

5. `check`
   - Obtiene telemetría actual
   - No modifica el estado
   - Retorna datos completos

## Simulación de Comportamiento

El simulador genera variaciones realistas en:

1. **Red**
   - Degradación gradual de señal WiFi
   - Fluctuaciones en velocidad de conexión
   - Variaciones en latencia
   - Conteo preciso de paquetes

2. **Hardware**
   - Patrones de uso de CPU
   - Temperatura basada en carga
   - Tendencias en uso de memoria

3. **Sensores**
   - Cambios graduales en temperatura
   - Variaciones lentas en humedad
   - Fluctuaciones menores en voltaje

4. **Geolocalización**
   - Variaciones dentro del rango de precisión WiFi (~50m)
   - Actualizaciones periódicas de coordenadas

## Notas Técnicas

1. **Conexión**
   - Comunicación bidireccional vía WebSocket
   - Heartbeat cada 250ms
   - Timeout de conexión: 20 segundos

2. **Datos**
   - Todos los timestamps incluyen zona horaria
   - Geolocalización vía WiFi (precisión ~50m)
   - Métricas de red combinan datos reales y simulados

3. **Estados**
   - Transiciones de estado registradas con timestamp
   - Bloqueo previene todos los comandos excepto `unblock` y `check`
   - Estado persiste entre reinicios

4. **Seguridad**
   - Registro de intentos de acceso no autorizado
   - Bloqueo remoto con razón personalizable
   - Monitoreo de estado de Chromium

## 5. Dependencias

```text
python-socketio>=5.1.0
psutil>=5.8.0
netifaces>=0.11.0
requests>=2.26.0
```

## 6. Notas Técnicas

1. **Zona Horaria**:
   - Cada kiosk maneja su propia zona horaria
   - Timestamps locales incluyen offset UTC
   - Sincronización mediante timestamp UTC

2. **Geolocalización**:
   - Obtenida vía WiFi
   - Incluye timestamp de última actualización
   - No requiere GPS

3. **Seguridad**:
   - Monitoreo de intentos de acceso no autorizados
   - Bloqueo preventivo ante amenazas
   - Registro detallado de eventos

4. **Comandos**:
   - Respuestas incluyen timestamp local
   - Confirmación de ejecución
   - Estado actual del kiosk

5. **Red**:
   - Monitoreo de conectividad
   - Detección de cambios de IP
   - Medición de calidad de conexión

