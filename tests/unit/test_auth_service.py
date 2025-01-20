# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import pytest
from app import create_app, db
from app.services.auth_service import AuthService
from app.models.user import User

@pytest.fixture
def app():
    """Fixture para crear una aplicación de prueba"""
    app = create_app('config.testing')
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

def test_user_registration(app):
    """Probar registro de usuario"""
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='testpassword'
    )
    
    assert user.username == 'testuser'
    assert user.email == 'test@example.com'
    assert user.check_password('testpassword') is True

def test_user_registration_duplicate_username(app):
    """Probar registro de usuario con nombre de usuario duplicado"""
    AuthService.register_user(
        username='testuser', 
        email='test1@example.com', 
        password='testpassword1'
    )
    
    with pytest.raises(ValueError, match="Nombre de usuario ya existe"):
        AuthService.register_user(
            username='testuser', 
            email='test2@example.com', 
            password='testpassword2'
        )

def test_user_registration_duplicate_email(app):
    """Probar registro de usuario con correo electrónico duplicado"""
    AuthService.register_user(
        username='testuser1', 
        email='test@example.com', 
        password='testpassword1'
    )
    
    with pytest.raises(ValueError, match="Correo electrónico ya registrado"):
        AuthService.register_user(
            username='testuser2', 
            email='test@example.com', 
            password='testpassword2'
        )

def test_user_authentication(app):
    """Probar autenticación de usuario"""
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='testpassword'
    )
    
    authenticated_user = AuthService.authenticate('testuser', 'testpassword')
    assert authenticated_user is not None
    assert authenticated_user.username == 'testuser'

def test_user_authentication_invalid_credentials(app):
    """Probar autenticación con credenciales inválidas"""
    AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='testpassword'
    )
    
    authenticated_user = AuthService.authenticate('testuser', 'wrongpassword')
    assert authenticated_user is None

def test_change_password(app):
    """Probar cambio de contraseña"""
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='oldpassword'
    )
    
    AuthService.change_password(user, 'oldpassword', 'newpassword')
    
    authenticated_user = AuthService.authenticate('testuser', 'newpassword')
    assert authenticated_user is not None

def test_change_password_invalid_old_password(app):
    """Probar cambio de contraseña con contraseña antigua incorrecta"""
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='oldpassword'
    )
    
    with pytest.raises(ValueError, match="Contraseña actual incorrecta"):
        AuthService.change_password(user, 'wrongoldpassword', 'newpassword') 