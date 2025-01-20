# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import pytest
from app import create_app, db
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
    user = User(username='testuser', email='test@example.com')
    user.set_password('testpassword')
    db.session.add(user)
    db.session.commit()
    return user

def test_create_kiosk(app, user):
    """Probar creación de kiosk"""
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    assert kiosk.name == 'Test Kiosk'
    assert kiosk.location == 'Test Location'
    assert kiosk.owner_id == user.id
    assert kiosk.status == 'inactive'
    assert kiosk.uuid is not None
    assert kiosk.cpu_model is not None
    assert kiosk.ram_total is not None
    assert kiosk.storage_total is not None
    assert kiosk.ip_address is not None

def test_get_kiosk_by_uuid(app, user):
    """Probar obtención de kiosk por UUID"""
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    retrieved_kiosk = KioskService.get_kiosk_by_uuid(kiosk.uuid)
    
    assert retrieved_kiosk is not None
    assert retrieved_kiosk.uuid == kiosk.uuid

def test_get_kiosks_by_owner(app, user):
    """Probar obtención de kiosks por propietario"""
    KioskService.create_kiosk(
        name='Test Kiosk 1', 
        location='Test Location 1', 
        owner_id=user.id
    )
    KioskService.create_kiosk(
        name='Test Kiosk 2', 
        location='Test Location 2', 
        owner_id=user.id
    )
    
    kiosks = KioskService.get_kiosks_by_owner(user.id)
    
    assert len(kiosks) == 2
    assert all(kiosk.owner_id == user.id for kiosk in kiosks)

def test_update_kiosk_status(app, user):
    """Probar actualización de estado de kiosk"""
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    updated_kiosk = KioskService.update_kiosk_status(kiosk.uuid, 'active')
    
    assert updated_kiosk.status == 'active'

def test_update_kiosk_hardware(app, user):
    """Probar actualización de información de hardware"""
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    updated_kiosk = KioskService.update_kiosk_hardware(
        kiosk.uuid, 
        cpu_model='Intel Core i7', 
        ram_total=16.0, 
        storage_total=512.0
    )
    
    assert updated_kiosk.cpu_model == 'Intel Core i7'
    assert updated_kiosk.ram_total == 16.0
    assert updated_kiosk.storage_total == 512.0

def test_update_kiosk_network(app, user):
    """Probar actualización de información de red"""
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    updated_kiosk = KioskService.update_kiosk_network(
        kiosk.uuid, 
        ip_address='192.168.1.100', 
        mac_address='00:11:22:33:44:55'
    )
    
    assert updated_kiosk.ip_address == '192.168.1.100'
    assert updated_kiosk.mac_address == '00:11:22:33:44:55'

def test_mark_kiosk_online_offline(app, user):
    """Probar marcado de kiosk como online/offline"""
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    online_kiosk = KioskService.mark_kiosk_online(kiosk.uuid)
    
    assert online_kiosk.status == 'active'
    assert online_kiosk.last_online is not None
    
    offline_kiosk = KioskService.mark_kiosk_offline(kiosk.uuid)
    
    assert offline_kiosk.status == 'inactive'

def test_delete_kiosk(app, user):
    """Probar eliminación de kiosk"""
    kiosk = KioskService.create_kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    result = KioskService.delete_kiosk(kiosk.uuid)
    
    assert result is True
    assert KioskService.get_kiosk_by_uuid(kiosk.uuid) is None 