# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
from unittest.mock import Mock, patch
from app.services.geolocation_service import GeolocationService
from app.models.kiosk import Kiosk

@pytest.fixture
def geolocation_service():
    """Fixture para el servicio de geolocalización"""
    return GeolocationService()

@pytest.fixture
def mock_kiosk():
    """Fixture para un kiosk de prueba"""
    kiosk = Mock(spec=Kiosk)
    kiosk.uuid = "test-uuid"
    return kiosk

def test_validate_coordinates_valid(geolocation_service):
    """Prueba validación de coordenadas válidas"""
    is_valid, error = geolocation_service.validate_coordinates(
        latitude=40.7128,
        longitude=-74.0060
    )
    assert is_valid is True
    assert error is None

def test_validate_coordinates_invalid_latitude(geolocation_service):
    """Prueba validación de latitud inválida"""
    is_valid, error = geolocation_service.validate_coordinates(
        latitude=91,
        longitude=-74.0060
    )
    assert is_valid is False
    assert "Latitud" in error

def test_validate_coordinates_invalid_longitude(geolocation_service):
    """Prueba validación de longitud inválida"""
    is_valid, error = geolocation_service.validate_coordinates(
        latitude=40.7128,
        longitude=181
    )
    assert is_valid is False
    assert "Longitud" in error

@patch('app.services.geolocation_service.requests.get')
def test_get_address_from_coordinates_success(mock_get, geolocation_service):
    """Prueba obtención exitosa de dirección desde coordenadas"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'display_name': 'Test Address, City, Country'
    }
    mock_get.return_value = mock_response

    address = geolocation_service.get_address_from_coordinates(
        latitude=40.7128,
        longitude=-74.0060
    )
    assert address == 'Test Address, City, Country'

@patch('app.services.geolocation_service.requests.get')
def test_get_address_from_coordinates_failure(mock_get, geolocation_service):
    """Prueba fallo en obtención de dirección"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    address = geolocation_service.get_address_from_coordinates(
        latitude=40.7128,
        longitude=-74.0060
    )
    assert address is None

@patch('app.services.geolocation_service.db.session.commit')
def test_update_kiosk_location_success(
    mock_commit,
    geolocation_service,
    mock_kiosk
):
    """Prueba actualización exitosa de ubicación"""
    with patch.object(
        geolocation_service,
        'get_address_from_coordinates',
        return_value='Test Address'
    ):
        success = geolocation_service.update_kiosk_location(
            kiosk=mock_kiosk,
            latitude=40.7128,
            longitude=-74.0060
        )
        
        assert success is True
        mock_kiosk.update_location.assert_called_once()
        mock_commit.assert_called_once()

@patch('app.services.geolocation_service.db.session.commit')
def test_update_kiosk_location_failure(
    mock_commit,
    geolocation_service,
    mock_kiosk
):
    """Prueba fallo en actualización de ubicación"""
    mock_commit.side_effect = Exception("Test error")
    
    success = geolocation_service.update_kiosk_location(
        kiosk=mock_kiosk,
        latitude=40.7128,
        longitude=-74.0060
    )
    
    assert success is False

@patch('app.services.geolocation_service.Kiosk.query')
def test_get_nearby_kiosks(mock_query, geolocation_service):
    """Prueba búsqueda de kiosks cercanos"""
    mock_kiosks = [Mock(spec=Kiosk) for _ in range(3)]
    mock_query.filter.return_value.params.return_value.all.return_value = mock_kiosks

    nearby = geolocation_service.get_nearby_kiosks(
        latitude=40.7128,
        longitude=-74.0060,
        radius_km=5.0
    )
    
    assert len(nearby) == 3
    mock_query.filter.assert_called_once() 