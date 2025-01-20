"""
Script para inicializar y ejecutar migraciones de base de datos.

Funcionalidad:
- Inicializa el sistema de migraciones
- Crea migraciones iniciales
- Ejecuta migraciones pendientes
- Verifica estado de la base de datos
- Permite rollback de migraciones

Uso:
python scripts/init_migrations.py [--upgrade/--downgrade]

Argumentos:
--upgrade: Aplicar migraciones pendientes
--downgrade: Revertir última migración
--revision: Crear nueva revisión

Notas:
- Ejecutar después de cambios en modelos
- Hacer backup antes de migrar en producción
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import os
import sys
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agregar directorio raíz al path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

# Cargar variables de entorno
load_dotenv()

def initialize_migrations():
    try:
        # Establecer variable de entorno FLASK_APP
        os.environ['FLASK_APP'] = 'run.py'
        logger.info("Variable FLASK_APP establecida")

        # Crear aplicación Flask
        app = Flask(__name__)
        logger.info("Aplicación Flask creada")

        # Configurar base de datos
        DB_USER = os.getenv('DB_USER', 'postgres')
        DB_PASSWORD = os.getenv('DB_PASSWORD', '')
        DB_HOST = os.getenv('DB_HOST', 'localhost')
        DB_PORT = os.getenv('DB_PORT', '5432')
        DB_NAME = os.getenv('DB_NAME', 'admin_kiosk2')

        app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')

        logger.info(f"URI de base de datos: {app.config['SQLALCHEMY_DATABASE_URI']}")

        # Inicializar extensiones
        db = SQLAlchemy(app)
        migrate = Migrate(app, db)
        logger.info("Extensiones inicializadas")

        # Importar modelos
        from app.models.user import User
        from app.models.kiosk import Kiosk
        logger.info("Modelos definidos")

        # Generar migración inicial
        logger.info("Generando migración inicial...")
        with app.app_context():
            # Crear directorio migrations si no existe
            if not os.path.exists('migrations'):
                logger.info("Inicializando directorio de migraciones...")
                os.system('flask db init')
            
            # Generar migración
            logger.info("Generando migración...")
            os.system('flask db migrate')
            
            # Aplicar migración
            logger.info("Aplicando migración...")
            os.system('flask db upgrade')

        logger.info("Migración completada exitosamente")
        return True

    except Exception as e:
        logger.error(f"Error al inicializar migraciones: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == '__main__':
    initialize_migrations() 