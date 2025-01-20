"""
Este script crea kiosks de prueba para desarrollo y testing.

Funcionalidad:
- Genera kiosks con datos aleatorios pero realistas
- Crea ubicaciones geográficas distribuidas
- Simula estados y métricas de los kiosks
- Útil para pruebas y desarrollo local

Uso:
python scripts/create_test_kiosks.py [número_de_kiosks]

Notas:
- Los datos generados son solo para pruebas
- No usar en producción
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import create_app, db
from app.models.kiosk import Kiosk, SensorData
from app.services.kiosk_service import KioskService
import random
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def create_test_kiosks(num_kiosks=5):
    """Crea kiosks de prueba con datos realistas"""
    try:
        logger.info(f"Creando {num_kiosks} kiosks de prueba...")
        
        locations = [
            "Sucursal Norte", "Sucursal Sur", "Sucursal Este",
            "Sucursal Oeste", "Sucursal Central", "Terminal A",
            "Terminal B", "Plaza Principal", "Centro Comercial",
            "Estación Central"
        ]
        
        # Coordenadas base (centro de la ciudad)
        base_lat = 19.4326
        base_lon = -99.1332
        
        for i in range(num_kiosks):
            # Generar ubicación aleatoria cercana al centro
            lat = base_lat + random.uniform(-0.1, 0.1)
            lon = base_lon + random.uniform(-0.1, 0.1)
            
            # Crear kiosk
            kiosk = KioskService.create_kiosk(
                name=f"Kiosk {i+1}",
                location=random.choice(locations)
            )
            
            # Actualizar ubicación
            kiosk.update_location(
                latitude=lat,
                longitude=lon,
                altitude=random.uniform(0, 100),
                accuracy=random.uniform(1, 10)
            )
            
            # Actualizar información de hardware
            KioskService.update_kiosk_status(
                kiosk.id,
                status='active',
                hardware_info={
                    'cpu_model': 'Intel Core i5-9400',
                    'ram_total': 8.0,
                    'storage_total': 256.0,
                    'ip_address': f"192.168.1.{random.randint(2, 254)}",
                    'mac_address': ':'.join(['%02x'%random.randint(0, 255) for _ in range(6)])
                }
            )
            
            # Generar datos de sensores históricos
            for j in range(24):  # Últimas 24 horas
                timestamp = datetime.utcnow() - timedelta(hours=j)
                KioskService.register_sensor_data(
                    kiosk.id,
                    cpu_usage=random.uniform(20, 80),
                    memory_usage=random.uniform(30, 70),
                    network_latency=random.uniform(5, 100)
                )
            
            logger.info(f"Kiosk {kiosk.name} creado exitosamente")
        
        logger.info("Creación de kiosks de prueba completada")
        
    except Exception as e:
        logger.error(f"Error creando kiosks de prueba: {str(e)}")
        raise

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        create_test_kiosks() 