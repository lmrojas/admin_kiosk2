# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import pytest
from app import create_app, db
from app.services.auth_service import AuthService
from app.services.kiosk_service import KioskService
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

@pytest.fixture
def user(app):
    """Crear usuario de prueba"""
    user = AuthService.register_user(
        username='testuser', 
        email='test@example.com', 
        password='testpassword'
    )
    return user

def test_kiosk_creation_and_retrieval(app, user):
    """Probar creación y recuperación de kiosk"""
    # Crear kiosk usando el servicio
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    # Recuperar kiosk por UUID
    retrieved_kiosk = KioskService.get_kiosk_by_uuid(kiosk.uuid)
    
    # Verificar detalles del kiosk
    assert retrieved_kiosk is not None
    assert retrieved_kiosk.name == 'Test Kiosk'
    assert retrieved_kiosk.location == 'Test Location'
    assert retrieved_kiosk.owner_id == user.id

def test_kiosk_status_workflow(app, user):
    """Probar flujo de cambios de estado de kiosk"""
    # Crear kiosk
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    # Estado inicial
    assert kiosk.status == 'inactive'
    
    # Marcar como online
    online_kiosk = KioskService.mark_kiosk_online(kiosk.uuid)
    assert online_kiosk.status == 'active'
    assert online_kiosk.last_online is not None
    
    # Marcar como offline
    offline_kiosk = KioskService.mark_kiosk_offline(kiosk.uuid)
    assert offline_kiosk.status == 'inactive'

def test_kiosk_hardware_update_workflow(app, user):
    """Probar flujo de actualización de hardware"""
    # Crear kiosk
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    # Actualizar información de hardware
    updated_kiosk = KioskService.update_kiosk_hardware(
        kiosk.uuid, 
        cpu_model='Intel Core i7', 
        ram_total=16.0, 
        storage_total=512.0
    )
    
    # Verificar actualización
    assert updated_kiosk.cpu_model == 'Intel Core i7'
    assert updated_kiosk.ram_total == 16.0
    assert updated_kiosk.storage_total == 512.0

def test_kiosk_network_update_workflow(app, user):
    """Probar flujo de actualización de red"""
    # Crear kiosk
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    # Actualizar información de red
    updated_kiosk = KioskService.update_kiosk_network(
        kiosk.uuid, 
        ip_address='192.168.1.100', 
        mac_address='00:11:22:33:44:55'
    )
    
    # Verificar actualización
    assert updated_kiosk.ip_address == '192.168.1.100'
    assert updated_kiosk.mac_address == '00:11:22:33:44:55'

def test_kiosk_deletion_workflow(app, user):
    """Probar flujo de eliminación de kiosk"""
    # Crear kiosk
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    # Verificar existencia inicial
    initial_kiosk = KioskService.get_kiosk_by_uuid(kiosk.uuid)
    assert initial_kiosk is not None
    
    # Eliminar kiosk
    deletion_result = KioskService.delete_kiosk(kiosk.uuid)
    
    # Verificar eliminación
    assert deletion_result is True
    deleted_kiosk = KioskService.get_kiosk_by_uuid(kiosk.uuid)
    assert deleted_kiosk is None

def test_multiple_kiosks_for_user(app, user):
    """Probar creación y recuperación de múltiples kiosks"""
    # Crear múltiples kiosks
    kiosk1 = KioskService.create_kiosk(
        name='Test Kiosk 1', 
        location='Location 1', 
        owner_id=user.id
    )
    kiosk2 = KioskService.create_kiosk(
        name='Test Kiosk 2', 
        location='Location 2', 
        owner_id=user.id
    )
    
    # Recuperar kiosks del usuario
    user_kiosks = KioskService.get_kiosks_by_owner(user.id)
    
    # Verificar múltiples kiosks
    assert len(user_kiosks) == 2
    kiosk_names = [kiosk.name for kiosk in user_kiosks]
    assert 'Test Kiosk 1' in kiosk_names
    assert 'Test Kiosk 2' in kiosk_names 