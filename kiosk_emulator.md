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
   - Actúa como un “kiosk real” con su ID/serial, ubicación, sensores, etc.
   - **Envío** de métricas al admin_kiosk (evento `kiosk_update`).
   - **Recepción** de acciones (`kiosk_command`): “bloquear”, “reiniciar”, “update”...
   - Su “spawner” (`kiosk_spawner.py`) crea n kiosks simultáneamente (ej.: 10).

**Objetivo**: Mantener la simulación y testing de kiosks **fuera** del sistema principal, que los ve como “datos reales”.

---

## 2. Kiosk App Independiente

### 2.1 ¿Qué es?
- Un **script/app** (`kiosk_app.py`) que corre **por fuera** de `admin_kiosk`.  
- Cada kiosk:
  1. **Identidad**: `kiosk_id`, `serial`.  
  2. **Ubicación asignada** vs. **ubicación real** (para generar discrepancias).  
  3. **Estados**: `online`, `offline`, `blocked`, `maintenance`...  
  4. **Sensores**: CPU, RAM, Disco, Temperatura, etc.  
  5. **WebSocket**: se conecta a `admin_kiosk` y le manda actualizaciones.

### 2.2 Separación de Lógica

- **Envío**: kiosk_app ⇒ admin_kiosk
  - `kiosk_update`: estado, sensores, ubicación real, etc.  
- **Recepción**: kiosk_app ⇐ admin_kiosk
  - `kiosk_command`: “reiniciar”, “bloquear”, “update”, etc.
  - El kiosk simula esos comportamientos.

### 2.3 Crear Múltiples Kiosks

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

class KioskConfig:
    def __init__(self, kiosk_id, serial, assigned_location, real_location):
        self.kiosk_id = kiosk_id
        self.serial = serial
        self.assigned_location = assigned_location
        self.real_location = real_location
        self.current_status = "offline"

# Creamos un cliente SocketIO (lado kiosk)
sio = socketio.Client()

@sio.event
def connect():
    print("[KioskApp] Conectado a admin_kiosk.")
    # Anunciar
    kiosk_data = {
        "kiosk_id": kiosk_config.kiosk_id,
        "serial": kiosk_config.serial
    }
    sio.emit('kiosk_join', kiosk_data)

@sio.event
def disconnect():
    print("[KioskApp] Desconectado de admin_kiosk.")

# === Lógica de RECEPCIÓN: admin_kiosk => kiosk_app
@sio.on('kiosk_command')
def on_kiosk_command(data):
    cmd = data.get("command")
    print(f"[KioskApp] Recibido comando: {cmd}")

    if cmd == "block":
        kiosk_config.current_status = "blocked"
        print("[KioskApp] Kiosk bloqueado, no enviará datos.")
    elif cmd == "reiniciar":
        print("[KioskApp] Reiniciando kiosk...")
        kiosk_config.current_status = "offline"
        time.sleep(3)
        kiosk_config.current_status = "online"
        print("[KioskApp] Reinicio completado.")
    elif cmd == "update":
        print("[KioskApp] Actualizando software kiosk...")
        time.sleep(2)
        print("[KioskApp] Actualización finalizada.")
    else:
        print(f"[KioskApp] Comando desconocido: {cmd}")

    # Notificar al admin que terminamos
    sio.emit('kiosk_command_ack', {
        "kiosk_id": kiosk_config.kiosk_id,
        "command": cmd,
        "status": "done"
    })

# === Lógica de ENVÍO: kiosk_app => admin_kiosk
def generate_sensor_data():
    """Genera datos aleatorios de CPU, RAM, etc."""
    return {
        "cpu_usage": random.uniform(10, 95),
        "ram_usage": random.uniform(20, 90),
        "disk_usage": random.uniform(30, 99),
        "temperature": random.uniform(15, 45),
        "network_latency": random.uniform(5, 300)
    }

def main_kiosk_loop(server_url):
    try:
        sio.connect(server_url)
    except Exception as e:
        print(f"[KioskApp] Error al conectar: {e}")
        return

    try:
        while True:
            if kiosk_config.current_status == "blocked":
                time.sleep(5)
                continue
            if kiosk_config.current_status == "offline":
                time.sleep(3)
                continue

            sensors = generate_sensor_data()
            # Simular cambios en ubicación real
            kiosk_config.real_location["lat"] += random.uniform(-0.00005, 0.00005)
            kiosk_config.real_location["lng"] += random.uniform(-0.00005, 0.00005)

            payload = {
                "kiosk_id": kiosk_config.kiosk_id,
                "serial": kiosk_config.serial,
                "status": kiosk_config.current_status,
                "assigned_location": kiosk_config.assigned_location,
                "real_location": kiosk_config.real_location,
                "sensors": sensors
            }

            print("[KioskApp] Enviando actualización:", payload)
            sio.emit('kiosk_update', payload)

            time.sleep(5)
    except KeyboardInterrupt:
        print("[KioskApp] Interrumpido por teclado.")
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
# Lanza multiples kiosks en paralelo
import subprocess
import sys

def spawn_kiosks(num_kiosks=10, server_url="http://localhost:5000"):
    processes = []
    for i in range(1, num_kiosks+1):
        kiosk_id = i
        serial = f"SIM-KIOSK-{i:03d}"
        cmd = [
            sys.executable, "kiosk_app.py",
            "--kiosk_id", str(kiosk_id),
            "--serial", serial,
            "--server_url", server_url
        ]
        proc = subprocess.Popen(cmd)
        processes.append(proc)
    return processes

if __name__ == "__main__":
    processes = spawn_kiosks(num_kiosks=10, server_url="http://localhost:5000")
    print(f"[Spawner] Lanzados {len(processes)} kiosks.")
```

> Por defecto, lanza **10** kiosks conectados a `http://localhost:5000`. Cada kiosk es un proceso que corre `kiosk_app.py` con sus propios argumentos.

---

## 5. Lógica en admin_kiosk (ej. SocketIO)

Dentro de `admin_kiosk`, podría haber un archivo `socket_handlers.py` (o integrado en `run.py`) con:

```python
# admin_kiosk/socket_handlers.py
from flask_socketio import SocketIO, emit

socketio = SocketIO(app)

@socketio.on('connect')
def on_connect():
    print("[Admin] Un kiosk se conectó (socket).")

@socketio.on('kiosk_join')
def on_kiosk_join(data):
    kiosk_id = data.get("kiosk_id")
    serial = data.get("serial")
    print(f"[Admin] kiosk_join => kiosk_id={kiosk_id}, serial={serial}")
    # Podrías registrar en BD, logs...

@socketio.on('kiosk_update')
def on_kiosk_update(data):
    kiosk_id = data.get("kiosk_id")
    sensors = data.get("sensors", {})
    # Guardar en DB, IA, logs...
    print(f"[Admin] kiosk_update => kiosk_id={kiosk_id}, sensors={sensors}")

def send_command_to_kiosk(kiosk_id, command):
    print(f"[Admin] Enviando comando '{command}' al kiosk {kiosk_id}")
    socketio.emit('kiosk_command', {
        "kiosk_id": kiosk_id,
        "command": command
    })
```

Para “bloquear kiosk 3”, harías:
```python
send_command_to_kiosk(3, "block")
```
El kiosk #3 recibe `'kiosk_command'` con `{"command":"block"}` y deja de emitir datos.

---

## 6. Conclusión

- **kiosk_app**:
  1. ENVÍA datos (`kiosk_update`) cada X segundos.
  2. RECIBE comandos (`kiosk_command`) y simula su efecto localmente.
- **kiosk_spawner.py**:
  - Lanza *n* kiosks (por defecto 10) para pruebas de carga, cada uno con un ID distinto.
- **admin_kiosk**:
  - Sólo ve datos “reales” via WebSocket.  
  - Puede mandar acciones a los kiosks.
  - No tiene simulación interna.

Resultado:  
- **Separación clara** entre la simulación (kiosk_app) y la lógica principal (admin_kiosk).  
- **Testing** con decenas/cientos de kiosks en paralelo.  
- **admin_kiosk** sigue siendo un sistema de MVT + IA, completamente ajeno a la simulación.

