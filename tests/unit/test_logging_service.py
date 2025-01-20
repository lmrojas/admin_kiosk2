# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.services.logging_service import LoggingService
from app import create_app

@pytest.fixture
def app():
    """Fixture que proporciona una aplicación Flask de prueba"""
    app = create_app('testing')
    app.config['TESTING'] = True
    app.config['LOG_LEVEL'] = 'DEBUG'
    
    # Crear directorio temporal para logs
    test_log_dir = os.path.join(app.root_path, 'logs_test')
    os.makedirs(test_log_dir, exist_ok=True)
    app.config['LOG_DIR'] = test_log_dir
    
    yield app
    
    # Limpiar logs después de las pruebas
    for file in os.listdir(test_log_dir):
        os.remove(os.path.join(test_log_dir, file))
    os.rmdir(test_log_dir)

@pytest.fixture
def logging_service(app):
    """Fixture que proporciona una instancia del servicio de logging"""
    with app.app_context():
        return LoggingService()

@pytest.fixture
def mock_request():
    """Fixture que proporciona un mock de request"""
    with patch('app.services.logging_service.request') as mock:
        mock.remote_addr = '127.0.0.1'
        mock.user_agent.string = 'Test Browser'
        mock.endpoint = 'test_endpoint'
        mock.method = 'GET'
        yield mock

def test_log_security_event(logging_service, mock_request, caplog):
    """Prueba el registro de eventos de seguridad"""
    event_type = 'login_attempt'
    details = {'username': 'test_user', 'success': True}
    
    logging_service.log_security_event(event_type, details)
    
    # Verificar que el log fue creado
    assert 'security' in caplog.text
    assert event_type in caplog.text
    assert 'test_user' in caplog.text

def test_log_audit_event(logging_service, mock_request, caplog):
    """Prueba el registro de eventos de auditoría"""
    user_id = 1
    action = 'create_kiosk'
    resource = 'kiosk'
    status = 'success'
    details = {'kiosk_id': 123}
    
    logging_service.log_audit_event(user_id, action, resource, status, details)
    
    # Verificar que el log fue creado
    assert 'audit' in caplog.text
    assert action in caplog.text
    assert str(user_id) in caplog.text
    assert str(details['kiosk_id']) in caplog.text

def test_log_error(logging_service, mock_request, caplog):
    """Prueba el registro de errores"""
    error = ValueError('Test error')
    context = {'function': 'test_function'}
    
    logging_service.log_error(error, context)
    
    # Verificar que el log fue creado
    assert 'ERROR' in caplog.text
    assert 'ValueError' in caplog.text
    assert 'Test error' in caplog.text
    assert 'test_function' in caplog.text

def test_get_logs_with_date_filter(logging_service, mock_request, tmp_path):
    """Prueba la obtención de logs filtrados por fecha"""
    # Crear algunos logs de prueba
    now = datetime.utcnow()
    log_entries = [
        {
            'timestamp': (now - timedelta(hours=2)).isoformat(),
            'event': 'old_event'
        },
        {
            'timestamp': now.isoformat(),
            'event': 'current_event'
        },
        {
            'timestamp': (now + timedelta(hours=2)).isoformat(),
            'event': 'future_event'
        }
    ]
    
    # Escribir logs en archivo
    log_file = os.path.join(logging_service.logger.handlers[0].baseFilename)
    with open(log_file, 'w') as f:
        for entry in log_entries:
            f.write(f"2024-01-01 12:00:00 - TEST - INFO - {json.dumps(entry)}\n")
    
    # Obtener logs filtrados
    start_date = now - timedelta(hours=1)
    end_date = now + timedelta(hours=1)
    filtered_logs = logging_service.get_logs('kiosk', start_date, end_date)
    
    assert len(filtered_logs) == 1
    assert filtered_logs[0]['event'] == 'current_event'

def test_log_rotation(logging_service, mock_request, caplog):
    """Prueba la rotación de archivos de log"""
    # Generar suficientes logs para causar rotación
    large_data = 'x' * 1024 * 1024  # 1MB
    for _ in range(15):  # Debería causar rotación (más de 10MB)
        logging_service.log_security_event('test', {'data': large_data})
    
    # Verificar que existen archivos de rotación
    log_dir = os.path.dirname(logging_service.logger.handlers[0].baseFilename)
    rotation_files = [f for f in os.listdir(log_dir) if f.endswith('.1')]
    assert len(rotation_files) > 0

def test_invalid_log_entry(logging_service, mock_request):
    """Prueba el manejo de entradas de log inválidas"""
    # Escribir una entrada inválida en el log
    log_file = os.path.join(logging_service.logger.handlers[0].baseFilename)
    with open(log_file, 'w') as f:
        f.write("Invalid JSON data\n")
    
    # Intentar leer los logs
    logs = logging_service.get_logs('kiosk', datetime.utcnow(), datetime.utcnow())
    assert len(logs) == 0  # No debería fallar, solo ignorar la entrada inválida 