# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

"""
Script para inicializar kiosks simulados en la base de datos.
Debe ejecutarse después de init_roles.py y antes de simulate_kiosks.py.
"""

from app import create_app, db
from app.models.user import User
from app.services.kiosk_service import KioskService
import logging
import random

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ubicaciones en Paraguay (las mismas que en simulate_kiosks.py)
LOCATIONS = [
    "Shopping del Sol", "Paseo La Galería", "Shopping Mariscal López",
    "Shopping Villa Morra", "Multiplaza", "Shopping Pinedo",
    "Terminal de Ómnibus", "Plaza de los Héroes", "Mercado 4",
    "Shopping San Lorenzo", "Plaza Uruguaya", "Mercado Municipal de Luque",
    "Shopping Mcal. López", "Plaza Infante Rivarola", "Costanera de Asunción",
    "Shopping París", "Plaza de la Democracia", "Terminal de Ciudad del Este",
    "Shopping China", "Mercado Municipal de Fernando", "Plaza Juan E. O'Leary",
    "Shopping Zuni", "Plaza Italia", "Mercado Central de Abasto",
    "Shopping Five Stars", "Plaza de las Américas", "Terminal de San Lorenzo",
    "Shopping Continental", "Plaza Uruguaya", "Mercado 4 de Asunción"
]

def init_kiosks():
    """Inicializa los kiosks simulados en la base de datos"""
    logger.info("Inicializando kiosks simulados...")
    
    # Obtener usuario admin
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        raise ValueError("Usuario admin no encontrado. Ejecute init_roles.py primero.")
    
    # Crear kiosks
    for i, location in enumerate(LOCATIONS):
        try:
            # Verificar si el kiosk ya existe
            kiosks = KioskService.get_all_kiosks()
            if any(k.name == f"Kiosk {location}" for k in kiosks):
                logger.info(f"Kiosk {location} ya existe, saltando...")
                continue
            
            # Crear nuevo kiosk
            kiosk = KioskService.create_kiosk(
                name=f"Kiosk {location}",
                location=location,
                owner_id=admin.id
            )
            
            # Actualizar con datos iniciales
            KioskService.update_kiosk(kiosk.id, {
                'status': 'active',
                'latitude': random.uniform(-25.3867, -25.2667),
                'longitude': random.uniform(-57.3333, -57.2833),
                'altitude': random.uniform(43, 250),
                'cpu_model': random.choice([
                    "Intel Core i3-10100", "Intel Core i5-10400",
                    "AMD Ryzen 3 3200G", "AMD Ryzen 5 3400G"
                ]),
                'ram_total': random.choice([4096, 8192, 16384]),
                'storage_total': random.choice([128, 256, 512])
            })
            
            logger.info(f"Kiosk {location} creado exitosamente")
            
        except Exception as e:
            logger.error(f"Error creando kiosk {location}: {str(e)}")
    
    logger.info("Inicialización de kiosks completada")

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        init_kiosks() 