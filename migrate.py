#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar migraciones de la base de datos.
"""

import os
import sys
from pathlib import Path
import logging
from flask_migrate import Migrate, upgrade

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_migrations():
    """Ejecuta las migraciones pendientes"""
    try:
        # Importar app y db
        from app import create_app, db
        from app.services.kiosk_service import KioskService
        
        # Crear aplicación
        app = create_app()
        
        # Configurar migración
        migrate = Migrate(app, db)
        
        with app.app_context():
            logger.info("Iniciando migraciones...")
            
            # Ejecutar migraciones pendientes
            upgrade()
            
            # Verificar estructura
            logger.info("Verificando estructura de tablas...")
            KioskService.verify_tables()
            
            logger.info("Migraciones completadas exitosamente")
            
    except Exception as e:
        logger.error(f"Error durante la migración: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    run_migrations()
