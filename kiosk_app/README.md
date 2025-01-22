# Kiosk App - Simulador de Kiosks

Este módulo simula el comportamiento de kiosks reales, permitiendo probar diferentes escenarios y situaciones sin necesidad de hardware físico.

## Estructura de Directorios

```
admin_kiosk_2/
└── kiosk_app/
    ├── db_sync.py
    ├── kiosk_app.py
    ├── anomaly_simulator.py
    ├── kiosk_client.py
    ├── kiosk_spawner.py
    ├── README.md
    └── kiosk_configs/        # Directorio de configuraciones
        ├── <uuid1>.json
        ├── <uuid2>.json
        └── summary.json
```

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Sincronizar con la Base de Datos

Ejecutar desde el directorio raíz del proyecto (admin_kiosk_2):

```bash
python -m kiosk_app.db_sync
```

Esto creará el directorio `kiosk_app/kiosk_configs/` y generará los archivos de configuración para cada kiosk activo en la BD.

### 2. Iniciar Simulación

#### a) Simular un Kiosk Individual
```bash
python -m kiosk_app.kiosk_client kiosk_app/kiosk_configs/<uuid>.json
```
Donde `<uuid>` es el identificador del kiosk que quieres simular.

#### b) Simular Múltiples Kiosks
```bash
python -m kiosk_app.kiosk_spawner --config-dir kiosk_app/kiosk_configs
```

Este comando:
1. Buscará el directorio `kiosk_configs` dentro del módulo kiosk_app
2. Cargará todas las configuraciones encontradas
3. Iniciará la simulación de todos los kiosks

### 3. Comportamientos Simulados

Los kiosks simulados:
- Envían datos de telemetría cada 5 segundos (CPU, memoria, red, etc.)
- Responden a comandos del servidor central
- Pueden simular diferentes tipos de anomalías:
  - Picos de CPU
  - Fugas de memoria
  - Latencia de red alta
  - Deriva en la ubicación
  - Pérdidas de conexión

### 4. Comandos Disponibles

El servidor puede enviar los siguientes comandos a los kiosks:

- `status_update`: Actualiza el estado del kiosk
- `get_telemetry`: Solicita datos de telemetría actuales
- `start_anomaly`: Inicia una simulación de anomalía específica
- `stop_anomaly`: Detiene una anomalía activa
- `random_anomaly`: Inicia una anomalía aleatoria

### 5. Variables de Entorno

Crear un archivo `.env` con:
```
DB_NAME=admin_kiosk2
DB_USER=postgres
DB_PASSWORD=tu_contraseña
DB_HOST=localhost
DB_PORT=5432
```

## Ejemplos de Uso

1. Iniciar un kiosk específico con servidor personalizado:
```bash
python -m kiosk_app.kiosk_client kiosk_app/kiosk_configs/abc123.json --server-url http://192.168.1.100:5000
```

2. Iniciar múltiples kiosks con configuración personalizada:
```bash
python -m kiosk_app.kiosk_spawner --config-dir custom_configs --server-url http://192.168.1.100:5000
```

## Notas

- Los kiosks simulados intentarán reconectarse automáticamente si pierden conexión
- Cada kiosk mantiene su propio estado y puede simular anomalías independientemente
- Los datos de telemetría son generados con variaciones realistas
- Las anomalías simuladas pueden ayudar a probar la detección de problemas en el sistema central 