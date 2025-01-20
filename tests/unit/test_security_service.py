# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.services.security_service import SecurityService
from app import create_app, db

@pytest.fixture
def app():
    """Fixture que proporciona una aplicación Flask de prueba"""
    app = create_app('testing')
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-key'
    app.config['JWT_SECRET_KEY'] = 'test-jwt-key'
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def security_service(app):
    """Fixture que proporciona una instancia del servicio de seguridad"""
    return SecurityService()

@pytest.fixture
def mock_redis():
    """Fixture que proporciona un mock de Redis"""
    with patch('redis.Redis') as mock:
        redis_client = MagicMock()
        mock.return_value = redis_client
        yield redis_client

def test_generate_jwt(security_service):
    """Prueba la generación de tokens JWT"""
    user_id = 1
    token = security_service.generate_jwt(user_id)
    
    assert token is not None
    assert isinstance(token, str)
    
    # Verificar el token
    payload = security_service.verify_jwt(token)
    assert payload is not None
    assert payload['user_id'] == user_id

def test_verify_jwt_invalid(security_service):
    """Prueba la verificación de tokens JWT inválidos"""
    invalid_token = 'invalid.token.here'
    payload = security_service.verify_jwt(invalid_token)
    assert payload is None

def test_verify_jwt_expired(security_service, app):
    """Prueba la verificación de tokens JWT expirados"""
    user_id = 1
    
    # Generar token que expira inmediatamente
    with patch('app.services.security_service.datetime') as mock_datetime:
        now = datetime.utcnow()
        mock_datetime.utcnow.return_value = now
        token = security_service.generate_jwt(user_id, {'exp': now})
    
    # Verificar después de la expiración
    with patch('app.services.security_service.datetime') as mock_datetime:
        mock_datetime.utcnow.return_value = now + timedelta(seconds=1)
        payload = security_service.verify_jwt(token)
        assert payload is None

def test_rate_limit_success(security_service, mock_redis):
    """Prueba el rate limiting cuando está dentro del límite"""
    mock_redis.get.return_value = None
    
    result = security_service.rate_limit('test_key', 10, 60)
    
    assert result is True
    mock_redis.setex.assert_called_once_with('test_key', 60, 1)

def test_rate_limit_exceeded(security_service, mock_redis):
    """Prueba el rate limiting cuando se excede el límite"""
    mock_redis.get.return_value = b'10'  # Límite alcanzado
    
    result = security_service.rate_limit('test_key', 10, 60)
    
    assert result is False
    mock_redis.incr.assert_not_called()

def test_rate_limit_increment(security_service, mock_redis):
    """Prueba el incremento del contador en rate limiting"""
    mock_redis.get.return_value = b'5'  # A mitad del límite
    
    result = security_service.rate_limit('test_key', 10, 60)
    
    assert result is True
    mock_redis.incr.assert_called_once_with('test_key')

def test_audit_log_success(security_service, caplog):
    """Prueba el registro de auditoría exitoso"""
    security_service.audit_log(
        user_id=1,
        action='test_action',
        resource='test_resource',
        details={'test': 'data'}
    )
    
    assert 'test_action' in caplog.text
    assert 'test_resource' in caplog.text
    assert 'success' in caplog.text

def test_audit_log_error(security_service, caplog):
    """Prueba el registro de auditoría con error"""
    mock_logger = MagicMock()
    mock_logger.info.side_effect = Exception('Test error')
    security_service.logger = mock_logger
    
    security_service.audit_log(
        user_id=1,
        action='test_action',
        resource='test_resource',
        details={'test': 'data'}
    )
    
    assert 'Error en audit log' in caplog.text

def test_rate_limit_redis_error(security_service, mock_redis):
    """Prueba el rate limiting cuando Redis falla"""
    mock_redis.get.side_effect = Exception('Redis error')
    
    result = security_service.rate_limit('test_key', 10, 60)
    
    # En caso de error, permitir el request
    assert result is True 