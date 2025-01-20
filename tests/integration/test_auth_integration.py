import pytest
from app import create_app, db
from app.services.auth_service import AuthService
from app.models.user import User
from app.models.kiosk import Kiosk

@pytest.fixture
def app():
    """Fixture para crear una aplicación de prueba"""
    app = create_app('config.testing')
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

def test_user_registration_and_kiosk_creation(app):
    """Probar registro de usuario y creación de kiosk"""
    # Registrar usuario
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='testpassword'
    )
    
    # Crear kiosk para el usuario
    kiosk = Kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    db.session.add(kiosk)
    db.session.commit()
    
    # Verificar registro de usuario
    assert user.username == 'testuser'
    assert user.email == 'test@example.com'
    
    # Verificar creación de kiosk
    assert kiosk.owner_id == user.id
    assert kiosk.name == 'Test Kiosk'
    assert kiosk.location == 'Test Location'

def test_user_authentication_with_kiosk(app):
    """Probar autenticación de usuario con kiosk asociado"""
    # Registrar usuario
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='testpassword'
    )
    
    # Crear kiosk para el usuario
    kiosk = Kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    db.session.add(kiosk)
    db.session.commit()
    
    # Autenticar usuario
    authenticated_user = AuthService.authenticate('testuser', 'testpassword')
    
    # Verificar autenticación
    assert authenticated_user is not None
    assert authenticated_user.username == 'testuser'
    
    # Verificar kiosks del usuario
    assert len(authenticated_user.kiosks) == 1
    assert authenticated_user.kiosks[0].name == 'Test Kiosk'

def test_user_password_change_and_authentication(app):
    """Probar cambio de contraseña y autenticación"""
    # Registrar usuario
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='oldpassword'
    )
    
    # Cambiar contraseña
    AuthService.change_password(user, 'oldpassword', 'newpassword')
    
    # Intentar autenticar con contraseña antigua (debe fallar)
    old_auth_user = AuthService.authenticate('testuser', 'oldpassword')
    assert old_auth_user is None
    
    # Autenticar con nueva contraseña
    new_auth_user = AuthService.authenticate('testuser', 'newpassword')
    assert new_auth_user is not None
    assert new_auth_user.username == 'testuser'

def test_multiple_kiosks_for_user(app):
    """Probar múltiples kiosks para un usuario"""
    # Registrar usuario
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='testpassword'
    )
    
    # Crear múltiples kiosks
    kiosk1 = Kiosk(
        name='Test Kiosk 1', 
        location='Location 1', 
        owner_id=user.id
    )
    kiosk2 = Kiosk(
        name='Test Kiosk 2', 
        location='Location 2', 
        owner_id=user.id
    )
    
    db.session.add(kiosk1)
    db.session.add(kiosk2)
    db.session.commit()
    
    # Verificar kiosks del usuario
    assert len(user.kiosks) == 2
    kiosk_names = [kiosk.name for kiosk in user.kiosks]
    assert 'Test Kiosk 1' in kiosk_names
    assert 'Test Kiosk 2' in kiosk_names 