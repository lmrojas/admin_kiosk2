# Admin Kiosk 2 - Simulador de Kiosks

Este proyecto implementa un simulador de kiosks que permite probar y validar el funcionamiento del sistema Admin Kiosk 2.

## Características Principales

- Simulación de múltiples kiosks en simultáneo
- Sincronización con base de datos PostgreSQL
- Comunicación WebSocket con servidor central
- Simulación de sensores y telemetría
- Manejo de comandos remotos
- Logging detallado de operaciones

## Requisitos

- Python 3.8+
- PostgreSQL
- Dependencias listadas en `requirements.txt`

## Estructura del Proyecto

```
admin_kiosk_2/
├── kiosk_app/
│   ├── __init__.py       # Inicializador del paquete
│   ├── kiosk_app.py      # Módulo principal de simulación
│   └── kiosk_configs/    # Configuraciones de kiosks
├── logs/                 # Archivos de log
├── migrations/          # Migraciones de base de datos
├── requirements.txt     # Dependencias del proyecto
├── .env                # Variables de entorno (no incluir en git)
└── README.md           # Este archivo
```

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd admin_kiosk_2
```

2. Crear entorno virtual:
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno en `.env`:
```ini
# Base de datos
DB_NAME=admin_kiosk2
DB_USER=postgres
DB_PASSWORD=tu_contraseña
DB_HOST=localhost
DB_PORT=5432

# Servidor WebSocket
WS_SERVER_URL=http://localhost:5000
```

## Uso

El simulador tiene dos modos de operación:

1. Sincronización con base de datos:
```bash
# Desde el directorio raíz del proyecto
python -m kiosk_app.kiosk_app --mode sync
```

2. Ejecución de simulación:
```bash
# Desde el directorio raíz del proyecto
python -m kiosk_app.kiosk_app --mode run
```

Opciones adicionales:
```bash
# Especificar URL del servidor
python -m kiosk_app.kiosk_app --mode run --server-url http://localhost:5000

# Ver ayuda
python -m kiosk_app.kiosk_app --help
```

## Clases Principales

### KioskConfig
Dataclass que almacena la configuración de cada kiosk:
- `serial`: Identificador único del kiosk
- `name`: Nombre descriptivo
- `location`: Diccionario con latitud, longitud y precisión
- `settings`: Configuraciones específicas del kiosk

### KioskSimulator
Simula un kiosk individual:
- Generación de telemetría (temperatura, humedad, etc.)
- Monitoreo de estado de sensores
- Procesamiento de comandos remotos
- Simulación de eventos y anomalías

### KioskManager
Gestiona múltiples kiosks:
- Conexión y reconexión WebSocket automática
- Registro y autenticación de kiosks
- Envío periódico de telemetría
- Manejo de eventos y comandos
- Sincronización con base de datos

## Logs

Los logs se guardan en el directorio `logs/` con el siguiente formato:
```
logs/
├── kiosk_simulation_YYYYMMDD_HHMMSS.log  # Logs de simulación
└── error_YYYYMMDD_HHMMSS.log             # Logs de errores
```

## Contribución

Por favor, sigue las guías de desarrollo en `cura.md` para contribuir al proyecto.

## Licencia

Este proyecto es privado y confidencial. 