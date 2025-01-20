# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import pytest
from flask_login import current_user
from app import create_app, db
from app.services.auth_service import AuthService
from app.services.kiosk_service import KioskService
from app.models.user import User
from app.models.kiosk import Kiosk

@pytest.fixture
def app():
    """Fixture para crear una aplicación de prueba E2E"""
    app = create_app('config.testing')
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    """Crear cliente de pruebas"""
    return app.test_client()

def test_complete_kiosk_management_workflow(app, client):
    """
    Prueba de flujo completo de gestión de kiosks:
    1. Registro de usuario
    2. Inicio de sesión
    3. Creación de kiosk
    4. Actualización de kiosk
    5. Verificación de kiosk
    6. Eliminación de kiosk
    """
    # 1. Registro de usuario
    registration_data = {
        'username': 'e2euser',
        'email': 'e2e@example.com',
        'password': 'securepassword123',
        'confirm_password': 'securepassword123'
    }
    
    # Registrar usuario
    user = AuthService.register_user(
        username=registration_data['username'],
        email=registration_data['email'],
        password=registration_data['password']
    )
    
    # 2. Inicio de sesión
    authenticated_user = AuthService.authenticate(
        registration_data['username'], 
        registration_data['password']
    )
    assert authenticated_user is not None
    
    # 3. Creación de kiosk
    kiosk_data = {
        'name': 'E2E Test Kiosk',
        'location': 'Test Location',
        'owner_id': user.id
    }
    
    kiosk = KioskService.create_kiosk(
        name=kiosk_data['name'],
        location=kiosk_data['location'],
        owner_id=kiosk_data['owner_id']
    )
    
    # Verificar creación de kiosk
    assert kiosk is not None
    assert kiosk.name == kiosk_data['name']
    assert kiosk.location == kiosk_data['location']
    assert kiosk.owner_id == user.id
    
    # 4. Actualización de kiosk
    updated_kiosk = KioskService.update_kiosk_hardware(
        kiosk.uuid, 
        cpu_model='Intel Core i5',
        ram_total=8.0,
        storage_total=256.0
    )
    
    # Verificar actualización de hardware
    assert updated_kiosk.cpu_model == 'Intel Core i5'
    assert updated_kiosk.ram_total == 8.0
    assert updated_kiosk.storage_total == 256.0
    
    # 5. Verificación de kiosk
    retrieved_kiosk = KioskService.get_kiosk_by_uuid(kiosk.uuid)
    assert retrieved_kiosk is not None
    assert retrieved_kiosk.name == kiosk_data['name']
    
    # Cambiar estado del kiosk
    online_kiosk = KioskService.mark_kiosk_online(kiosk.uuid)
    assert online_kiosk.status == 'active'
    
    # 6. Eliminación de kiosk
    deletion_result = KioskService.delete_kiosk(kiosk.uuid)
    assert deletion_result is True
    
    # Verificar eliminación
    deleted_kiosk = KioskService.get_kiosk_by_uuid(kiosk.uuid)
    assert deleted_kiosk is None

def test_multiple_kiosk_management(app, client):
    """
    Prueba de gestión de múltiples kiosks por un usuario
    """
    # Registro de usuario
    user = AuthService.register_user(
        username='multiuser',
        email='multi@example.com',
        password='multipassword123'
    )
    
    # Crear múltiples kiosks
    kiosk_names = ['Kiosk Alpha', 'Kiosk Beta', 'Kiosk Gamma']
    created_kiosks = []
    
    for name in kiosk_names:
        kiosk = KioskService.create_kiosk(
            name=name,
            location=f'Location of {name}',
            owner_id=user.id
        )
        created_kiosks.append(kiosk)
    
    # Verificar creación de múltiples kiosks
    user_kiosks = KioskService.get_kiosks_by_owner(user.id)
    assert len(user_kiosks) == 3
    
    # Verificar nombres de kiosks
    kiosk_names_retrieved = [kiosk.name for kiosk in user_kiosks]
    assert set(kiosk_names_retrieved) == set(kiosk_names)

def test_kiosk_network_configuration(app, client):
    """
    Prueba de configuración de red de kiosk
    """
    # Registro de usuario
    user = AuthService.register_user(
        username='networkuser',
        email='network@example.com',
        password='networkpassword123'
    )
    
    # Crear kiosk
    kiosk = KioskService.create_kiosk(
        name='Network Kiosk',
        location='Network Test Location',
        owner_id=user.id
    )
    
    # Configurar información de red
    network_config = {
        'ip_address': '192.168.100.50',
        'mac_address': '00:1A:2B:3C:4D:5E'
    }
    
    updated_kiosk = KioskService.update_kiosk_network(
        kiosk.uuid,
        ip_address=network_config['ip_address'],
        mac_address=network_config['mac_address']
    )
    
    # Verificar configuración de red
    assert updated_kiosk.ip_address == network_config['ip_address']
    assert updated_kiosk.mac_address == network_config['mac_address'] 