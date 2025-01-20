# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
import json
import os
from datetime import datetime, timedelta
from flask import current_app
from app.services.logging_service import LoggingService
from werkzeug.test import TestResponse

@pytest.fixture
def logging_service(app):
    """Fixture que proporciona una instancia del servicio de logging para pruebas"""
    with app.test_request_context():
        service = LoggingService()
        yield service

def test_security_log_creation_and_retrieval(app, logging_service):
    """Prueba la creación y recuperación de logs de seguridad"""
    with app.test_request_context('/', environ_base={'REMOTE_ADDR': '127.0.0.1'}):
        # Crear evento de seguridad
        event_type = 'test_login'
        details = {'username': 'test@example.com', 'success': True}
        
        logging_service.log_security_event(
            event_type=event_type,
            details=details,
            level='INFO'
        )
        
        # Recuperar logs
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(minutes=5)
        logs = logging_service.get_logs('security', start_date, end_date)
        
        assert len(logs) > 0
        latest_log = logs[0]
        assert latest_log['event_type'] == event_type
        assert latest_log['details']['username'] == details['username']
        assert latest_log['ip'] == '127.0.0.1'

def test_audit_log_creation_and_retrieval(app, logging_service):
    """Prueba la creación y recuperación de logs de auditoría"""
    with app.test_request_context('/', environ_base={'REMOTE_ADDR': '127.0.0.1'}):
        # Crear evento de auditoría
        user_id = 1
        action = 'test_create'
        resource = 'kiosk'
        status = 'success'
        details = {'kiosk_id': 123}
        
        logging_service.log_audit_event(
            user_id=user_id,
            action=action,
            resource=resource,
            status=status,
            details=details
        )
        
        # Recuperar logs
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(minutes=5)
        logs = logging_service.get_logs('audit', start_date, end_date)
        
        assert len(logs) > 0
        latest_log = logs[0]
        assert latest_log['user_id'] == user_id
        assert latest_log['action'] == action
        assert latest_log['resource'] == resource
        assert latest_log['status'] == status
        assert latest_log['details']['kiosk_id'] == details['kiosk_id']

def test_error_log_creation_and_retrieval(app, logging_service):
    """Prueba la creación y recuperación de logs de error"""
    with app.test_request_context('/test', method='POST', environ_base={'REMOTE_ADDR': '127.0.0.1'}):
        # Crear log de error
        try:
            raise ValueError("Test error message")
        except Exception as e:
            logging_service.log_error(
                error=e,
                context={'test_function': 'test_error_logging'}
            )
        
        # Recuperar logs
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(minutes=5)
        logs = logging_service.get_logs('error', start_date, end_date)
        
        assert len(logs) > 0
        latest_log = logs[0]
        assert latest_log['error_type'] == 'ValueError'
        assert latest_log['error_message'] == 'Test error message'
        assert latest_log['context']['test_function'] == 'test_error_logging'
        assert latest_log['method'] == 'POST'
        assert latest_log['endpoint'] == 'test'

def test_log_rotation(app, logging_service):
    """Prueba la rotación de archivos de log"""
    with app.test_request_context():
        # Generar suficientes logs para causar rotación
        large_data = 'x' * 1024 * 1024  # 1MB de datos
        for _ in range(15):  # Debería causar rotación (más de 10MB)
            logging_service.log_security_event(
                event_type='test_rotation',
                details={'data': large_data}
            )
        
        # Verificar que existen archivos de rotación
        log_dir = os.path.dirname(logging_service.logger.handlers[0].baseFilename)
        rotation_files = [f for f in os.listdir(log_dir) if f.endswith('.1')]
        assert len(rotation_files) > 0

def test_log_filtering(app, logging_service):
    """Prueba el filtrado de logs por fecha"""
    with app.test_request_context():
        # Crear logs con diferentes fechas
        dates = [
            datetime.utcnow() - timedelta(days=2),
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow()
        ]
        
        for date in dates:
            logging_service.log_security_event(
                event_type=f'test_event_{date.day}',
                details={'date': date.isoformat()}
            )
        
        # Filtrar logs del último día
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        logs = logging_service.get_logs('security', start_date, end_date)
        
        assert len(logs) >= 1
        for log in logs:
            log_date = datetime.fromisoformat(log['timestamp'])
            assert start_date <= log_date <= end_date

def test_invalid_log_handling(app, logging_service):
    """Prueba el manejo de entradas de log inválidas"""
    with app.test_request_context():
        # Intentar crear un log con datos inválidos
        with pytest.raises(Exception):
            logging_service.log_security_event(
                event_type=None,
                details=None
            )
        
        # Verificar que el sistema sigue funcionando
        logging_service.log_security_event(
            event_type='test_recovery',
            details={'status': 'ok'}
        )
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(minutes=5)
        logs = logging_service.get_logs('security', start_date, end_date)
        
        assert len(logs) > 0
        assert any(log['event_type'] == 'test_recovery' for log in logs) 