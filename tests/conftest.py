# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import os
import sys
import pytest
from pathlib import Path

# Agregar directorio raíz al path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from app import create_app, db

@pytest.fixture
def app():
    """Fixture que proporciona una aplicación Flask de prueba"""
    # Configurar base de datos de prueba
    os.environ['DB_USER'] = 'postgres'
    os.environ['DB_PASSWORD'] = 'postgres'
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_PORT'] = '5432'
    os.environ['DB_NAME'] = 'admin_kiosk2_test'
    
    # Crear aplicación con configuración de prueba
    app = create_app('config.default.TestingConfig')
    
    # Establecer contexto de aplicación
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    """Fixture que proporciona un cliente de prueba"""
    return app.test_client() 