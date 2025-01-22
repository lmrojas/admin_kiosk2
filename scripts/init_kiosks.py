# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

"""
Script para inicializar la estructura de la base de datos para kiosks.
Este script NO crea kiosks - solo prepara la estructura.
Los kiosks deben darse de alta desde el sistema y luego conectarse vía WebSocket usando su número de serie.

Debe ejecutarse después de init_roles.py.
"""

import os
import sys
from pathlib import Path

# Agregar directorio raíz al path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from app import create_app, db
from app.models.user import User
from app.services.kiosk_service import KioskService
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_kiosks():
    """
    Inicializa la estructura de la base de datos para kiosks.
    No crea ningún kiosk - solo verifica que la estructura esté lista.
    """
    logger.info("Verificando estructura para kiosks...")
    
    try:
        # Verificar que existe el usuario admin
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            raise ValueError("Usuario admin no encontrado. Ejecute init_roles.py primero.")
        
        # Verificar que las tablas están creadas
        KioskService.verify_tables()
        
        logger.info("Estructura de base de datos para kiosks verificada correctamente")
            
    except Exception as e:
        logger.error(f"Error verificando estructura de kiosks: {str(e)}")
        raise

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        init_kiosks() 