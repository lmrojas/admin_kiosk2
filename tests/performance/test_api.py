"""
Tests de rendimiento para endpoints de la API.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import pytest
import json
from flask import url_for
from app.models import User, Kiosk
from app.services.auth_service import AuthService

@pytest.fixture(scope="module")
def auth_token(test_client, db_session):
    """Fixture para obtener un token de autenticación."""
    auth_service = AuthService(db_session)
    
    # Crear usuario de prueba
    user = User(
        username="test_api_user",
        email="test_api@example.com",
        role_name="ADMIN"
    )
    user.set_password("test_password")
    db_session.add(user)
    db_session.commit()
    
    token = auth_service.authenticate("test_api_user", "test_password")
    return token

@pytest.fixture(scope="module")
def test_data(db_session):
    """Fixture para crear datos de prueba."""
    # Crear kiosks de prueba
    kiosks = []
    for i in range(50):
        kiosk = Kiosk(
            name=f"API Test Kiosk {i}",
            location=f"API Test Location {i}",
            status="active",
            health_score=95.0
        )
        kiosks.append(kiosk)
    
    db_session.bulk_save_objects(kiosks)
    db_session.commit()
    
    yield
    
    # Limpiar datos
    db_session.query(Kiosk).filter(Kiosk.name.like("API Test%")).delete()
    db_session.commit()

def test_get_kiosks_performance(benchmark, test_client, auth_token, test_data):
    """Test de rendimiento para obtener lista de kiosks."""
    def get_kiosks():
        return test_client.get(
            url_for('api.list_kiosks'),
            headers={'Authorization': f'Bearer {auth_token}'}
        )
    
    response = benchmark(get_kiosks)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['kiosks']) >= 50

def test_get_kiosk_details_performance(benchmark, test_client, auth_token, test_data, db_session):
    """Test de rendimiento para obtener detalles de un kiosk."""
    kiosk = db_session.query(Kiosk).filter(Kiosk.name.like("API Test%")).first()
    
    def get_kiosk_details():
        return test_client.get(
            url_for('api.get_kiosk', kiosk_id=kiosk.id),
            headers={'Authorization': f'Bearer {auth_token}'}
        )
    
    response = benchmark(get_kiosk_details)
    assert response.status_code == 200

def test_create_kiosk_performance(benchmark, test_client, auth_token):
    """Test de rendimiento para crear kiosks."""
    def create_kiosk():
        return test_client.post(
            url_for('api.create_kiosk'),
            headers={
                'Authorization': f'Bearer {auth_token}',
                'Content-Type': 'application/json'
            },
            data=json.dumps({
                'name': 'Performance Test Kiosk',
                'location': 'Test Location',
                'status': 'active',
                'health_score': 95.0
            })
        )
    
    response = benchmark(create_kiosk)
    assert response.status_code == 201

def test_update_kiosk_performance(benchmark, test_client, auth_token, db_session):
    """Test de rendimiento para actualizar kiosks."""
    kiosk = db_session.query(Kiosk).filter(Kiosk.name.like("API Test%")).first()
    
    def update_kiosk():
        return test_client.put(
            url_for('api.update_kiosk', kiosk_id=kiosk.id),
            headers={
                'Authorization': f'Bearer {auth_token}',
                'Content-Type': 'application/json'
            },
            data=json.dumps({
                'status': 'inactive',
                'health_score': 85.0
            })
        )
    
    response = benchmark(update_kiosk)
    assert response.status_code == 200

def test_concurrent_requests_performance(benchmark, test_client, auth_token):
    """Test de rendimiento para peticiones concurrentes."""
    def concurrent_requests():
        responses = []
        # Realizar múltiples peticiones
        for _ in range(10):
            responses.append(
                test_client.get(
                    url_for('api.list_kiosks'),
                    headers={'Authorization': f'Bearer {auth_token}'}
                )
            )
        return responses
    
    responses = benchmark(concurrent_requests)
    assert all(r.status_code == 200 for r in responses)

def test_search_kiosks_performance(benchmark, test_client, auth_token, test_data):
    """Test de rendimiento para búsqueda de kiosks."""
    def search_kiosks():
        return test_client.get(
            url_for('api.search_kiosks', query='API Test'),
            headers={'Authorization': f'Bearer {auth_token}'}
        )
    
    response = benchmark(search_kiosks)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['results']) > 0 