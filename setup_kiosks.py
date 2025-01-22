import os
import sys
import json
import uuid
import random
import subprocess
from datetime import datetime
from pathlib import Path
from app import create_app, db
from app.models.kiosk import Kiosk

# Configuración
NUM_KIOSKS = 30
KIOSK_APP_PATH = "kiosk_app/kiosk_app.py"

class KioskSetup:
    def __init__(self):
        """Inicializa la configuración de kiosks"""
        self.app = create_app()
        self.kiosks = []
        
    def setup_db(self):
        """Configura la conexión a la base de datos"""
        try:
            with self.app.app_context():
                db.create_all()
            print("✓ Base de datos configurada")
        except Exception as e:
            print(f"✗ Error configurando base de datos: {e}")
            sys.exit(1)
            
    def generate_kiosk_data(self):
        """Genera datos para 30 kiosks con ubicaciones diferentes"""
        locations = [
            {"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
            {"name": "Córdoba", "lat": -31.4201, "lon": -64.1888},
            {"name": "Rosario", "lat": -32.9468, "lon": -60.6393},
            {"name": "Mendoza", "lat": -32.8908, "lon": -68.8272},
            {"name": "San Miguel de Tucumán", "lat": -26.8083, "lon": -65.2176},
            {"name": "La Plata", "lat": -34.9215, "lon": -57.9545},
            {"name": "Mar del Plata", "lat": -38.0055, "lon": -57.5426},
            {"name": "Salta", "lat": -24.7829, "lon": -65.4232},
            {"name": "Santa Fe", "lat": -31.6333, "lon": -60.7000},
            {"name": "San Juan", "lat": -31.5375, "lon": -68.5364}
        ]
        
        for i in range(NUM_KIOSKS):
            location = random.choice(locations)
            # Añadir variación a las coordenadas para kiosks en la misma ciudad
            lat_variation = random.uniform(-0.01, 0.01)
            lon_variation = random.uniform(-0.01, 0.01)
            
            kiosk = {
                "uuid": str(uuid.uuid4()),
                "name": f"Kiosk_{i+1}_{location['name']}",
                "location": location['name'],
                "latitude": location['lat'] + lat_variation,
                "longitude": location['lon'] + lon_variation,
                "status": "active",
                "created_at": datetime.utcnow()
            }
            self.kiosks.append(kiosk)
        
        print(f"✓ Generados datos para {NUM_KIOSKS} kiosks")
            
    def register_kiosks(self):
        """Registra los kiosks en la base de datos"""
        try:
            with self.app.app_context():
                for kiosk_data in self.kiosks:
                    kiosk = Kiosk(
                        uuid=kiosk_data['uuid'],
                        name=kiosk_data['name'],
                        location=kiosk_data['location'],
                        latitude=kiosk_data['latitude'],
                        longitude=kiosk_data['longitude'],
                        status=kiosk_data['status'],
                        created_at=kiosk_data['created_at']
                    )
                    db.session.add(kiosk)
                db.session.commit()
            print("✓ Kiosks registrados en la base de datos")
        except Exception as e:
            print(f"✗ Error registrando kiosks: {e}")
            sys.exit(1)
            
    def save_kiosk_configs(self):
        """Guarda las configuraciones de los kiosks para la aplicación emuladora"""
        config_dir = Path("kiosk_app/kiosk_configs")
        config_dir.mkdir(exist_ok=True)
        
        for kiosk in self.kiosks:
            config_file = config_dir / f"{kiosk['uuid']}.json"
            with open(config_file, 'w') as f:
                json.dump(kiosk, f, indent=4, default=str)
        
        print("✓ Configuraciones de kiosks guardadas")
            
    def run(self):
        """Ejecuta el proceso completo de configuración"""
        print("\nIniciando configuración de kiosks...")
        self.setup_db()
        self.generate_kiosk_data()
        self.register_kiosks()
        self.save_kiosk_configs()
        print("\n✓ Configuración completada exitosamente\n")

if __name__ == '__main__':
    setup = KioskSetup()
    setup.run() 