"""
Script para registrar kiosks en el sistema central.
Sigue el modelo MVT+S usando los servicios existentes.
"""
import os
import json
import uuid
import random
import hashlib
import socket
import psutil
from datetime import datetime
from app import create_app, db
from app.services.kiosk_service import KioskService

# Crear la aplicación
app = create_app()
app.app_context().push()

# Configuración
KIOSKS_CONFIG_DIR = os.path.join('kiosk_app', 'kiosk_configs')
os.makedirs(KIOSKS_CONFIG_DIR, exist_ok=True)

def register_kiosks(num_kiosks=30):
    """Registra kiosks en el sistema central usando KioskService"""
    print("Iniciando registro de 30 kiosks...")
    
    locations = [
        {"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
        {"name": "Córdoba", "lat": -31.4201, "lon": -64.1888},
        {"name": "Rosario", "lat": -32.9468, "lon": -60.6393},
        {"name": "Mendoza", "lat": -32.8908, "lon": -68.8272},
        {"name": "San Miguel de Tucumán", "lat": -26.8083, "lon": -65.2176}
    ]
    
    registered = 0
    for i in range(num_kiosks):
        try:
            location = locations[i % len(locations)]
            lat_variation = random.uniform(-0.01, 0.01)
            lon_variation = random.uniform(-0.01, 0.01)
            
            # Generar UUID y credenciales
            kiosk_uuid = str(uuid.uuid4())
            credentials = f"{kiosk_uuid}:{datetime.utcnow().isoformat()}"
            credentials_hash = hashlib.sha256(credentials.encode()).hexdigest()
            
            # Obtener información de hardware
            hardware_info = {
                'cpu_model': 'Simulated CPU',
                'ram_total': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
                'storage_total': psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # GB
                'ip_address': socket.gethostbyname(socket.gethostname()),
                'mac_address': uuid.getnode().to_bytes(6, 'big').hex(':')
            }
            
            # Crear kiosk usando el servicio existente
            kiosk = KioskService.create_kiosk(
                name=f"Kiosk_{i+1}_{location['name']}",
                location=location['name']
            )
            
            # Actualizar información adicional
            KioskService.update_kiosk_status(
                kiosk.id,
                'inactive',
                hardware_info=hardware_info
            )
            
            # Guardar configuración para el kiosk_app (simulación)
            config = {
                "serial": kiosk.uuid,
                "name": kiosk.name,
                "location": {
                    "lat": location['lat'] + lat_variation,
                    "lon": location['lon'] + lon_variation,
                    "city": location['name']
                },
                "status": "active",
                "credentials": credentials,
                "hardware": hardware_info,
                "capabilities": {
                    'sensors': ['temperature', 'humidity', 'cpu', 'memory', 'network'],
                    'features': ['remote_control', 'telemetry', 'diagnostics']
                },
                "created_at": datetime.utcnow().isoformat()
            }
            
            config_path = os.path.join(KIOSKS_CONFIG_DIR, f"{kiosk.uuid}.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            registered += 1
            print(f"✓ Kiosk {kiosk.name} registrado exitosamente")
            
        except Exception as e:
            print(f"Error registrando kiosk {i+1}: {str(e)}")
            db.session.rollback()
            
    print(f"Registro completado. {registered} kiosks registrados.")
    print(f"Configuraciones guardadas en: {KIOSKS_CONFIG_DIR}")

if __name__ == "__main__":
    register_kiosks()