"""
Configuración para tests de seguridad.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
from app import create_app
from app.models import db
from app.models.user import User
import os

@pytest.fixture(scope='session')
def app():
    """Fixture que proporciona la aplicación Flask para testing."""
    app = create_app('testing')
    
    # Configurar la app para testing
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': True,
        'SECURITY_PASSWORD_SALT': 'test-salt'
    })
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture(scope='session')
def client(app):
    """Fixture que proporciona un cliente de test."""
    return app.test_client()

@pytest.fixture(scope='session')
def test_user(app):
    """Fixture que crea un usuario de prueba."""
    with app.app_context():
        user = User(
            email='test_pentest@example.com',
            username='test_pentest',
            password='Test123!@#'
        )
        db.session.add(user)
        db.session.commit()
        return user 