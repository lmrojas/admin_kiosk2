A continuación tienes la **versión final extendida** de la **Guía Definitiva** para un **Sistema de Kiosks** con Flask + IA (denominado `admin_kiosk`), acompañada de una **estructura** completa que incluye un **microservicio de kiosks** (`kiosk_app`) y su *spawner* para la creación y simulación de múltiples kiosks en paralelo.

---

# Guía Definitiva: admin_kiosk + kiosk_app

## 0. Estructura Global de Archivos

```text
admin_kiosk/
│
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── kiosk_ai_service.py
│   │   ├── kiosk_service.py
│   │   └── ...
│   ├── blueprints/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   └── kiosk.py
│   ├── templates/
│   │   ├── base.html
│   │   └── login.html
│   └── utils/
│       └── __init__.py
│
├── config/
│   └── default.py
│
├── migrations/
├── scripts/
│   ├── export_structure.py
│   ├── train_ai_model.py
│   ├── generate_synthetic_data.py
│   ├── auto_retrain_pipeline.py
│   └── ...
│
├── tests/
│   ├── unit/
│   └── integration/
├── requirements.txt
├── run.py
├── README.md
└── project_custom_structure.txt

kiosk_app/
│
├── kiosk_app.py         # Lógica principal de UN kiosk
├── kiosk_spawner.py     # Script para lanzar varios kiosks
├── requirements.txt     # Dependencias (p.ej. `python-socketio`)
└── README.md            # Documentación de kiosk_app
```

> - `admin_kiosk`: El **sistema principal** con Flask + IA.  
> - `kiosk_app`: **Aplicación aparte** que simula uno o más kiosks, conectándose vía WebSocket a `admin_kiosk`.

---

## 1. Visión General

1. **admin_kiosk**:
   - Recibe datos de kiosks (reales o simulados) a través de API o WebSocket.
   - Procesa alertas, IA, logs, vistas web.
   - **No** contiene lógica de simulación.

2. **kiosk_app**:
   - Actúa como un "kiosk real" con su ID/serial, ubicación, sensores, etc.
   - **Envío** de métricas al admin_kiosk (evento `kiosk_update`).
   - **Recepción** de acciones (`kiosk_command`): "bloquear", "reiniciar", "update"...
   - Su "spawner" (`kiosk_spawner.py`) crea n kiosks simultáneamente (ej.: 10).

**Objetivo**: Mantener la simulación y testing de kiosks **fuera** del sistema principal, que los ve como "datos reales".

---

## 2. Kiosk App Independiente

### 2.1 ¿Qué es?
- Un **script/app** (`kiosk_app.py`) que corre **por fuera** de `admin_kiosk`.  
- Cada kiosk:
  1. **Identidad**: `kiosk_id`, `serial` (debe estar registrado previamente).  
  2. **Validación**: Verifica credenciales y registro en sistema central.
  3. **Estados**: `online`, `offline`, `blocked`, `maintenance`...  
  4. **Sensores**: 
     - Ambientales: Temperatura, Humedad, Calidad del aire
     - Hardware: CPU, RAM, Disco, Red
     - Periféricos: Impresora, Lector de tarjetas, etc.
  5. **WebSocket**: Comunicación bidireccional con `admin_kiosk`.

### 2.2 Comunicación Bidireccional

#### Validación Inicial
```python
@sio.event
def connect():
    # 1. Validar credenciales y registro previo
    if not validate_kiosk_credentials():
        logger.error(f"Kiosk {kiosk_config.serial} no autorizado")
        return False
    
    # 2. Verificar registro en sistema central
    if not check_kiosk_registration():
        logger.error(f"Kiosk {kiosk_config.serial} no registrado")
        return False
        
    # 3. Anunciar conexión exitosa
    sio.emit('kiosk_join', {
        'kiosk_id': kiosk_config.kiosk_id,
        'serial': kiosk_config.serial,
        'capabilities': get_kiosk_capabilities()
    })
```

#### Envío de Datos (kiosk → admin)
```python
def generate_sensor_data():
    return {
        # Datos básicos
        "timestamp": datetime.utcnow().isoformat(),
        "kiosk_id": kiosk_config.kiosk_id,
        "serial": kiosk_config.serial,
        
        # Sensores ambientales
        "environmental": {
            "temperature": random.uniform(18, 30),
            "humidity": random.uniform(30, 70),
            "air_quality": random.uniform(0, 100)
        },
        
        # Hardware
        "hardware": {
            "cpu_temp": random.uniform(35, 75),
            "cpu_usage": random.uniform(10, 95),
            "ram_usage": random.uniform(20, 90),
            "disk_usage": random.uniform(30, 99)
        },
        
        # Periféricos
        "peripherals": {
            "printer_status": random.choice(["ready", "low_ink", "paper_jam", "offline"]),
            "cash_acceptor": random.choice(["ready", "full", "error"]),
            "card_reader": random.choice(["ready", "error"])
        },
        
        # Ubicación y red
        "location": {
            "lat": kiosk_config.real_location["lat"],
            "lng": kiosk_config.real_location["lng"],
            "accuracy": random.uniform(1, 10)
        },
        "network": {
            "latency": random.uniform(5, 300),
            "signal_strength": random.uniform(-90, -30)
        }
    }
```

#### Recepción de Comandos (admin → kiosk)
```python
@sio.on('kiosk_command')
def handle_command(data):
    cmd = data.get('command')
    params = data.get('params', {})
    
    response = {
        "kiosk_id": kiosk_config.kiosk_id,
        "command": cmd,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "processing"
    }
    
    try:
        if cmd == "block":
            kiosk_config.current_status = "blocked"
            response["status"] = "success"
            
        elif cmd == "restart":
            kiosk_config.current_status = "restarting"
            time.sleep(3)  # Simular reinicio
            kiosk_config.current_status = "online"
            response["status"] = "success"
            
        elif cmd == "update":
            kiosk_config.current_status = "updating"
            time.sleep(2)  # Simular actualización
            kiosk_config.current_status = "online"
            response["status"] = "success"
            
        elif cmd == "maintenance":
            kiosk_config.current_status = "maintenance"
            response["status"] = "success"
            
        else:
            response["status"] = "error"
            response["error"] = f"Comando desconocido: {cmd}"
            
    except Exception as e:
        response["status"] = "error"
        response["error"] = str(e)
        
    finally:
        # Confirmar ejecución del comando
        sio.emit('kiosk_command_ack', response)
```

### 2.3 Flujo de Datos y Estados

1. **Conexión Inicial**:
   - Validar credenciales y registro
   - Anunciar capacidades y estado

2. **Envío Periódico**:
   - Datos de sensores cada 5s
   - Estado de periféricos
   - Ubicación y red

3. **Recepción de Comandos**:
   - Bloqueo/Desbloqueo
   - Reinicio
   - Actualización
   - Mantenimiento

4. **Confirmaciones**:
   - ACK de comandos recibidos
   - Estado de ejecución
   - Errores o problemas

### 2.4 Crear Múltiples Kiosks

- Con `kiosk_spawner.py`, lanzamos *n* instancias de `kiosk_app.py`.
- Por defecto, *n=10*.
- Así **probamos** carga y escalabilidad.  

---

## 3. Ejemplo de `kiosk_app.py`

```python
# kiosk_app.py
# EL CÓDIGO DE ESTE ARCHIVO ES INDEPENDIENTE DEL SISTEMA admin_kiosk
# Y SOLO INTERACTÚA VÍA WEBSOCKET.

import time
import random
import socketio
import argparse
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KioskApp')

class KioskConfig:
    def __init__(self, kiosk_id, serial, assigned_location, real_location):
        self.kiosk_id = kiosk_id
        self.serial = serial
        self.assigned_location = assigned_location
        self.real_location = real_location
        self.current_status = "offline"
        self.capabilities = {
            "sensors": ["temperature", "humidity", "air_quality"],
            "peripherals": ["printer", "card_reader", "cash_acceptor"],
            "features": ["gps", "network_monitoring"]
        }

def validate_kiosk_credentials():
    """Validar credenciales del kiosk con el sistema central"""
    try:
        # Aquí iría la lógica real de validación
        return True
    except Exception as e:
        logger.error(f"Error en validación: {e}")
        return False

def check_kiosk_registration():
    """Verificar registro previo en sistema central"""
    try:
        # Aquí iría la lógica real de verificación
        return True
    except Exception as e:
        logger.error(f"Error en verificación: {e}")
        return False

def get_kiosk_capabilities():
    """Obtener capacidades del kiosk"""
    return kiosk_config.capabilities

# Creamos un cliente SocketIO (lado kiosk)
sio = socketio.Client()

@sio.event
def connect():
    logger.info("[KioskApp] Conectado a admin_kiosk.")
    
    # 1. Validar credenciales y registro
    if not validate_kiosk_credentials():
        logger.error(f"Kiosk {kiosk_config.serial} no autorizado")
        return False
    
    if not check_kiosk_registration():
        logger.error(f"Kiosk {kiosk_config.serial} no registrado")
        return False
    
    # 2. Anunciar conexión exitosa
    kiosk_data = {
        "kiosk_id": kiosk_config.kiosk_id,
        "serial": kiosk_config.serial,
        "capabilities": get_kiosk_capabilities()
    }
    sio.emit('kiosk_join', kiosk_data)

@sio.event
def disconnect():
    logger.info("[KioskApp] Desconectado de admin_kiosk.")
    kiosk_config.current_status = "offline"

@sio.on('kiosk_command')
def handle_command(data):
    cmd = data.get('command')
    params = data.get('params', {})
    
    response = {
        "kiosk_id": kiosk_config.kiosk_id,
        "command": cmd,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "processing"
    }
    
    try:
        if cmd == "block":
            kiosk_config.current_status = "blocked"
            response["status"] = "success"
            
        elif cmd == "restart":
            kiosk_config.current_status = "restarting"
            time.sleep(3)  # Simular reinicio
            kiosk_config.current_status = "online"
            response["status"] = "success"
            
        elif cmd == "update":
            kiosk_config.current_status = "updating"
            time.sleep(2)  # Simular actualización
            kiosk_config.current_status = "online"
            response["status"] = "success"
            
        elif cmd == "maintenance":
            kiosk_config.current_status = "maintenance"
            response["status"] = "success"
            
        else:
            response["status"] = "error"
            response["error"] = f"Comando desconocido: {cmd}"
            
    except Exception as e:
        response["status"] = "error"
        response["error"] = str(e)
        logger.error(f"Error ejecutando comando {cmd}: {e}")
        
    finally:
        sio.emit('kiosk_command_ack', response)

def generate_sensor_data():
    """Genera datos enriquecidos de sensores y estado"""
    return {
        # Datos básicos
        "timestamp": datetime.utcnow().isoformat(),
        "kiosk_id": kiosk_config.kiosk_id,
        "serial": kiosk_config.serial,
        
        # Sensores ambientales
        "environmental": {
            "temperature": random.uniform(18, 30),
            "humidity": random.uniform(30, 70),
            "air_quality": random.uniform(0, 100)
        },
        
        # Hardware
        "hardware": {
            "cpu_temp": random.uniform(35, 75),
            "cpu_usage": random.uniform(10, 95),
            "ram_usage": random.uniform(20, 90),
            "disk_usage": random.uniform(30, 99)
        },
        
        # Periféricos
        "peripherals": {
            "printer_status": random.choice(["ready", "low_ink", "paper_jam", "offline"]),
            "cash_acceptor": random.choice(["ready", "full", "error"]),
            "card_reader": random.choice(["ready", "error"])
        },
        
        # Ubicación y red
        "location": {
            "lat": kiosk_config.real_location["lat"],
            "lng": kiosk_config.real_location["lng"],
            "accuracy": random.uniform(1, 10)
        },
        "network": {
            "latency": random.uniform(5, 300),
            "signal_strength": random.uniform(-90, -30)
        }
    }

def main_kiosk_loop(server_url):
    """Bucle principal del kiosk"""
    try:
        sio.connect(server_url)
    except Exception as e:
        logger.error(f"Error al conectar: {e}")
        return

    try:
        while True:
            if kiosk_config.current_status == "blocked":
                logger.info("Kiosk bloqueado, esperando...")
                time.sleep(5)
                continue
                
            if kiosk_config.current_status == "offline":
                logger.info("Kiosk offline, intentando reconectar...")
                time.sleep(3)
                continue

            # Generar y enviar datos
            data = generate_sensor_data()
            logger.debug(f"Enviando actualización: {data}")
            sio.emit('kiosk_update', data)
            
            time.sleep(5)  # Intervalo de actualización
            
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario")
    except Exception as e:
        logger.error(f"Error en bucle principal: {e}")
    finally:
        if sio.connected:
            sio.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kiosk_id", type=int, default=1)
    parser.add_argument("--serial", type=str, default="SIM-KIOSK-001")
    parser.add_argument("--server_url", type=str, default="http://localhost:5000")
    args = parser.parse_args()

    kiosk_config = KioskConfig(
        kiosk_id=args.kiosk_id,
        serial=args.serial,
        assigned_location={"lat": -34.6037, "lng": -58.3816},
        real_location={"lat": -34.6037, "lng": -58.3816}
    )
    kiosk_config.current_status = "online"

    main_kiosk_loop(args.server_url)
```

### Notas
- Envía `kiosk_update` cada 5s.  
- Recibe `kiosk_command` y actúa en consecuencia (bloquear, reiniciar…).

---

## 4. Script `kiosk_spawner.py`

```python
# kiosk_spawner.py
# Lanza múltiples kiosks en paralelo con validación
import subprocess
import sys
import time
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KioskSpawner')

def load_kiosk_configs(config_file="kiosk_configs.json"):
    """Cargar configuraciones predefinidas de kiosks"""
    try:
        with open(config_file) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Archivo {config_file} no encontrado, usando configuración por defecto")
        return {}

def spawn_kiosks(num_kiosks=10, server_url="http://localhost:5000", configs=None):
    """Lanzar múltiples kiosks con validación"""
    processes = []
    
    for i in range(1, num_kiosks + 1):
        kiosk_id = i
        serial = f"SIM-KIOSK-{i:03d}"
        
        # Usar configuración específica si existe
        config = configs.get(str(i), {}) if configs else {}
        
        cmd = [
            sys.executable, "kiosk_app.py",
            "--kiosk_id", str(kiosk_id),
            "--serial", serial,
            "--server_url", server_url
        ]
        
        # Agregar configuraciones adicionales
        for key, value in config.items():
            cmd.extend([f"--{key}", str(value)])
        
        try:
            proc = subprocess.Popen(cmd)
            processes.append(proc)
            logger.info(f"Lanzado kiosk {kiosk_id} (serial: {serial})")
            time.sleep(1)  # Esperar entre lanzamientos
        except Exception as e:
            logger.error(f"Error lanzando kiosk {kiosk_id}: {e}")
    
    return processes

if __name__ == "__main__":
    configs = load_kiosk_configs()
    processes = spawn_kiosks(num_kiosks=10, server_url="http://localhost:5000", configs=configs)
    logger.info(f"Lanzados {len(processes)} kiosks")
```

> Por defecto, lanza **10** kiosks conectados a `http://localhost:5000`. Cada kiosk es un proceso que corre `kiosk_app.py` con sus propios argumentos.

---

## 5. Lógica en admin_kiosk

```python
# admin_kiosk/socket_handlers.py
from flask_socketio import SocketIO, emit
from datetime import datetime
import logging

logger = logging.getLogger('AdminKiosk')
socketio = SocketIO(app)

@socketio.on('connect')
def on_connect():
    """Manejo de nueva conexión de kiosk"""
    logger.info("[Admin] Nueva conexión de kiosk")

@socketio.on('kiosk_join')
def on_kiosk_join(data):
    """Validación y registro de nuevo kiosk"""
    kiosk_id = data.get("kiosk_id")
    serial = data.get("serial")
    capabilities = data.get("capabilities", {})
    
    logger.info(f"[Admin] Kiosk {kiosk_id} (serial: {serial}) intentando unirse")
    
    # Verificar registro previo
    if not kiosk_service.is_registered(serial):
        logger.warning(f"Kiosk {serial} no registrado")
        return False
    
    # Registrar capacidades y estado
    kiosk_service.update_kiosk_capabilities(kiosk_id, capabilities)
    kiosk_service.mark_kiosk_online(kiosk_id)
    
    logger.info(f"[Admin] Kiosk {kiosk_id} conectado exitosamente")

@socketio.on('kiosk_update')
def on_kiosk_update(data):
    """Procesar actualización de datos del kiosk"""
    kiosk_id = data.get("kiosk_id")
    timestamp = data.get("timestamp")
    
    try:
        # Almacenar datos en BD
        kiosk_service.store_kiosk_data(data)
        
        # Procesar con IA si corresponde
        if kiosk_ai_service.should_process(data):
            kiosk_ai_service.process_data(data)
        
        # Verificar alertas
        alerts = alert_service.check_alerts(data)
        if alerts:
            emit('kiosk_alerts', alerts, broadcast=True)
            
    except Exception as e:
        logger.error(f"Error procesando datos de kiosk {kiosk_id}: {e}")

@socketio.on('kiosk_command_ack')
def on_command_ack(data):
    """Procesar confirmación de comando"""
    kiosk_id = data.get("kiosk_id")
    command = data.get("command")
    status = data.get("status")
    
    logger.info(f"[Admin] ACK de kiosk {kiosk_id}: {command} => {status}")
    
    if status == "error":
        error_msg = data.get("error", "Error desconocido")
        logger.error(f"Error en kiosk {kiosk_id}: {error_msg}")

def send_command_to_kiosk(kiosk_id, command, params=None):
    """Enviar comando a un kiosk específico"""
    logger.info(f"[Admin] Enviando comando '{command}' al kiosk {kiosk_id}")
    
    socketio.emit('kiosk_command', {
        "kiosk_id": kiosk_id,
        "command": command,
        "params": params or {},
        "timestamp": datetime.utcnow().isoformat()
    })
```

## 6. Conclusión

- **Validación y Seguridad**:
  1. Verificación de credenciales y registro previo
  2. Capacidades anunciadas al conectar
  3. Confirmación de comandos (ACK)

- **Datos Enriquecidos**:
  1. Sensores ambientales y hardware
  2. Estado de periféricos
  3. Ubicación y calidad de red

- **Comunicación Bidireccional**:
  1. Kiosk → Admin: Datos y estados
  2. Admin → Kiosk: Comandos y configuración
  3. Sistema de confirmaciones

- **Monitoreo y Control**:
  1. Logging detallado
  2. Manejo de errores
  3. Alertas en tiempo real

Resultado:
- Sistema robusto y seguro
- Datos completos para análisis
- Control granular de kiosks
- Escalabilidad probada

