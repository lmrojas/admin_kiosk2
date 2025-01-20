# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import pytest
from app import create_app, db
from app.models.kiosk import Kiosk
from app.models.user import User
from datetime import datetime, timedelta

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

def test_kiosk_creation(app, user):
    """Probar creación de kiosk"""
    kiosk = Kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    db.session.add(kiosk)
    db.session.commit()
    
    assert kiosk.name == 'Test Kiosk'
    assert kiosk.location == 'Test Location'
    assert kiosk.owner_id == user.id
    assert kiosk.status == 'inactive'
    assert kiosk.uuid is not None

def test_kiosk_status_update(app, user):
    """Probar actualización de estado de kiosk"""
    kiosk = Kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    db.session.add(kiosk)
    db.session.commit()
    
    kiosk.update_status('active')
    
    assert kiosk.status == 'active'
    assert kiosk.updated_at is not None

def test_kiosk_hardware_update(app, user):
    """Probar actualización de información de hardware"""
    kiosk = Kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    db.session.add(kiosk)
    db.session.commit()
    
    kiosk.update_hardware_info(
        cpu_model='Intel Core i7', 
        ram_total=16.0, 
        storage_total=512.0
    )
    
    assert kiosk.cpu_model == 'Intel Core i7'
    assert kiosk.ram_total == 16.0
    assert kiosk.storage_total == 512.0

def test_kiosk_network_update(app, user):
    """Probar actualización de información de red"""
    kiosk = Kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    db.session.add(kiosk)
    db.session.commit()
    
    kiosk.update_network_info(
        ip_address='192.168.1.100', 
        mac_address='00:11:22:33:44:55'
    )
    
    assert kiosk.ip_address == '192.168.1.100'
    assert kiosk.mac_address == '00:11:22:33:44:55'

def test_kiosk_online_offline(app, user):
    """Probar marcado de kiosk como online/offline"""
    kiosk = Kiosk(
        name='Test Kiosk', 
        location='Test Location', 
        owner_id=user.id
    )
    
    db.session.add(kiosk)
    db.session.commit()
    
    kiosk.mark_online()
    
    assert kiosk.status == 'active'
    assert kiosk.last_online is not None
    
    kiosk.mark_offline()
    
    assert kiosk.status == 'inactive' 