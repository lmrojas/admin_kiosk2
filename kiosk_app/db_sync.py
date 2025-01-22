"""
Script para sincronizar kiosks desde la base de datos y generar configuraciones.
"""
import os
import json
import logging
import psycopg2
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Obtiene conexión a la base de datos usando variables de entorno"""
    load_dotenv()
    return psycopg2.connect(
        dbname=os.getenv('DB_NAME', 'admin_kiosk2'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432')
    )

def get_kiosks_from_db() -> List[Dict[str, Any]]:
    """Obtiene todos los kiosks activos de la base de datos"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    uuid, name, location, status, 
                    latitude, longitude, reported_latitude, reported_longitude,
                    cpu_model, ram_total, storage_total, 
                    ip_address, mac_address
                FROM kiosks 
                WHERE status != 'deleted'
            """)
            columns = [desc[0] for desc in cur.description]
            kiosks = []
            for row in cur.fetchall():
                kiosk = dict(zip(columns, row))
                kiosks.append(kiosk)
            return kiosks

def generate_kiosk_config(kiosk: Dict[str, Any]) -> Dict[str, Any]:
    """Genera configuración para un kiosk"""
    return {
        'serial': str(kiosk['uuid']),
        'name': kiosk['name'],
        'location': kiosk['location'],
        'status': kiosk['status'],
        'coordinates': {
            'assigned': {
                'latitude': float(kiosk['latitude']) if kiosk['latitude'] else None,
                'longitude': float(kiosk['longitude']) if kiosk['longitude'] else None
            },
            'reported': {
                'latitude': float(kiosk['reported_latitude']) if kiosk['reported_latitude'] else None,
                'longitude': float(kiosk['reported_longitude']) if kiosk['reported_longitude'] else None
            }
        },
        'hardware': {
            'cpu_model': kiosk['cpu_model'],
            'ram_total': float(kiosk['ram_total']) if kiosk['ram_total'] else None,
            'storage_total': float(kiosk['storage_total']) if kiosk['storage_total'] else None,
            'ip_address': kiosk['ip_address'],
            'mac_address': kiosk['mac_address']
        }
    }

def sync_kiosk_configs():
    """Sincroniza las configuraciones de kiosks con la base de datos"""
    try:
        # Mantener las configuraciones dentro del módulo kiosk_app
        config_dir = os.path.join(os.path.dirname(__file__), 'kiosk_configs')
        
        # Crear directorio si no existe
        os.makedirs(config_dir, exist_ok=True)
        logger.info(f"Usando directorio de configuración: {config_dir}")
        
        # Obtener kiosks de la BD
        kiosks = get_kiosks_from_db()
        logger.info(f"Obtenidos {len(kiosks)} kiosks de la base de datos")
        
        # Limpiar configuraciones existentes
        for file in os.listdir(config_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(config_dir, file))
        logger.info("Configuraciones existentes eliminadas")
        
        # Generar nuevas configuraciones
        for kiosk in kiosks:
            config = generate_kiosk_config(kiosk)
            filename = f"{kiosk['uuid']}.json"
            filepath = os.path.join(config_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
                
        logger.info(f"Generadas {len(kiosks)} configuraciones nuevas")
        
        # Generar resumen
        summary = {
            'last_sync': datetime.utcnow().isoformat(),
            'kiosk_count': len(kiosks)
        }
        with open(os.path.join(config_dir, 'kiosks_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        logger.info("Sincronización completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante la sincronización: {str(e)}")
        raise

if __name__ == '__main__':
    sync_kiosk_configs() 