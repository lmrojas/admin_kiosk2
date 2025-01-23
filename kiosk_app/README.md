# Kiosk App - Emulador de Kiosk

Módulo para simular kiosks que envían datos de telemetría vía WebSocket.

## Instalación

1. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

El módulo consta de dos scripts principales:

### 1. kiosk_app.py
Contiene la lógica base del kiosk para obtener datos del sistema y procesar comandos.

### 2. kiosk_simulator_runner.py 
Script principal para ejecutar la simulación. Este es el que debes ejecutar:

```bash
python kiosk_app/kiosk_simulator_runner.py
```

El script:
- Conecta con los kiosks registrados en el sistema central
- Envía datos de telemetría cada 5 segundos incluyendo:
  - Estado del sistema (OS, memoria, CPU)
  - Métricas de red
  - Datos de sensores
  - Información de seguridad
  - Geolocalización
- Procesa comandos recibidos (restart, update, block/unblock)

## Datos Enviados

La telemetría incluye:

- Identificación
  - UUID del kiosk
  - Nombre
  - Datos de red (IP, MAC)

- Estado del Sistema
  - Nombre y versión del OS
  - Uptime
  - Estado actual
  - Estado de Chromium
  - Zona horaria local

- Hardware
  - Uso de CPU y temperatura
  - Memoria
  - Disco

- Sensores
  - Temperatura ambiente
  - Humedad
  - Voltaje
  - Estado de puerta

- Seguridad
  - Últimos accesos no autorizados
  - Razón de bloqueo

- Geolocalización
  - Latitud/Longitud
  - Timestamp

## Comandos Soportados

- restart: Reinicia el kiosk
- update: Actualiza el software
- block: Bloquea el kiosk
- unblock: Desbloquea el kiosk
- check: Verifica estado actual

## Notas Técnicas

- La geolocalización se obtiene vía WiFi
- Los datos de sensores son simulados con variaciones realistas
- El estado se mantiene consistente entre reconexiones
- Todos los timestamps incluyen zona horaria

## Estructura

```
kiosk_app/
├── kiosk_app.py           # Clase principal del kiosk
├── kiosk_behavior_simulator.py  # Simulador de comportamiento
├── requirements.txt       # Dependencias del módulo
└── README.md             # Este archivo
``` 