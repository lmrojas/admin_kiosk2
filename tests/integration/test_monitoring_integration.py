# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
from datetime import datetime, timedelta
from app.services.monitoring_service import MonitoringService
from app.services.logging_service import LoggingService

@pytest.fixture
def monitoring_service(app):
    """Fixture que proporciona una instancia del servicio de monitoreo"""
    with app.test_request_context():
        return MonitoringService()

@pytest.fixture
def logging_service(app):
    """Fixture que proporciona una instancia del servicio de logging"""
    with app.test_request_context():
        return LoggingService()

def test_failed_login_alert(app, monitoring_service, logging_service):
    """Prueba la generación de alertas por intentos fallidos de login"""
    with app.test_request_context('/', environ_base={'REMOTE_ADDR': '127.0.0.1'}):
        # Generar intentos fallidos de login
        for _ in range(6):
            logging_service.log_security_event(
                event_type='login_attempt',
                details={'username': 'test@example.com', 'success': False}
            )
        
        # Verificar alertas
        alerts = monitoring_service.monitor_system()
        
        assert len(alerts) > 0
        alert = next(a for a in alerts if a['name'] == 'failed_login_attempts')
        assert alert['severity'] == 'high'
        assert 'email' in alert['channels']
        assert 'slack' in alert['channels']

def test_critical_error_alert(app, monitoring_service, logging_service):
    """Prueba la generación de alertas por errores críticos"""
    with app.test_request_context('/test', method='POST', environ_base={'REMOTE_ADDR': '127.0.0.1'}):
        # Generar error crítico
        try:
            raise ConnectionError("Database connection failed")
        except Exception as e:
            logging_service.log_error(
                error=e,
                context={'operation': 'database_connect'}
            )
        
        # Verificar alertas
        alerts = monitoring_service.monitor_system()
        
        assert len(alerts) > 0
        alert = next(a for a in alerts if a['name'] == 'critical_errors')
        assert alert['severity'] == 'critical'
        assert all(channel in alert['channels'] for channel in ['email', 'slack', 'sms'])

def test_suspicious_activity_alert(app, monitoring_service, logging_service):
    """Prueba la generación de alertas por actividad sospechosa"""
    with app.test_request_context('/', environ_base={'REMOTE_ADDR': '127.0.0.1'}):
        # Generar eventos de actividad sospechosa
        for _ in range(4):
            logging_service.log_security_event(
                event_type='unauthorized_access',
                details={'resource': 'admin_panel', 'user_id': 123}
            )
        
        # Verificar alertas
        alerts = monitoring_service.monitor_system()
        
        assert len(alerts) > 0
        alert = next(a for a in alerts if a['name'] == 'suspicious_activity')
        assert alert['severity'] == 'high'
        assert all(channel in alert['channels'] for channel in ['email', 'slack'])

def test_resource_modification_alert(app, monitoring_service, logging_service):
    """Prueba la generación de alertas por modificaciones excesivas de recursos"""
    with app.test_request_context('/', environ_base={'REMOTE_ADDR': '127.0.0.1'}):
        # Generar eventos de modificación
        for i in range(11):
            logging_service.log_audit_event(
                user_id=1,
                action='update',
                resource='kiosk',
                status='success',
                details={'kiosk_id': i}
            )
        
        # Verificar alertas
        alerts = monitoring_service.monitor_system()
        
        assert len(alerts) > 0
        alert = next(a for a in alerts if a['name'] == 'resource_modification')
        assert alert['severity'] == 'medium'
        assert 'email' in alert['channels']

def test_system_metrics(app, monitoring_service, logging_service):
    """Prueba la obtención de métricas del sistema"""
    with app.test_request_context():
        # Generar algunos eventos
        logging_service.log_security_event(
            event_type='login_attempt',
            details={'username': 'test@example.com', 'success': False}
        )
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            logging_service.log_error(error=e, context={})
        
        logging_service.log_audit_event(
            user_id=1,
            action='create',
            resource='kiosk',
            status='success',
            details={}
        )
        
        # Obtener métricas
        metrics = monitoring_service.get_system_metrics()
        
        assert 'last_hour' in metrics
        assert metrics['last_hour']['security_events'] >= 1
        assert metrics['last_hour']['errors'] >= 1
        assert metrics['last_hour']['audit_events'] >= 1
        assert metrics['last_hour']['failed_logins'] >= 1

def test_alerts_history(app, monitoring_service, logging_service):
    """Prueba la obtención del historial de alertas"""
    with app.test_request_context():
        # Generar algunas alertas
        for _ in range(6):
            logging_service.log_security_event(
                event_type='login_attempt',
                details={'username': 'test@example.com', 'success': False}
            )
        
        # Asegurar que se generen las alertas
        monitoring_service.monitor_system()
        
        # Obtener historial
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=1)
        alerts = monitoring_service.get_alerts_history(
            start_date=start_date,
            end_date=end_date,
            severity='high'
        )
        
        assert len(alerts) > 0
        assert all(alert['severity'] == 'high' for alert in alerts) 