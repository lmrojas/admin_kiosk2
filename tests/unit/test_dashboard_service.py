# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.services.dashboard_service import DashboardService

@pytest.fixture
def mock_services():
    """Fixture que proporciona mocks de los servicios dependientes"""
    with patch('app.services.dashboard_service.LoggingService') as mock_logging, \
         patch('app.services.dashboard_service.BackupService') as mock_backup, \
         patch('app.services.dashboard_service.MonitoringService') as mock_monitoring:
        
        # Mock de LoggingService
        mock_logging_instance = MagicMock()
        mock_logging_instance.get_log_count.return_value = 1000
        mock_logging_instance.get_error_count.return_value = 50
        mock_logging_instance.get_warning_count.return_value = 100
        mock_logging_instance.get_failed_login_count.return_value = 10
        mock_logging_instance.get_suspicious_activity_count.return_value = 5
        mock_logging_instance.get_daily_summary.return_value = [
            {'date': '2024-01-01', 'total': 100, 'errors': 5}
        ]
        mock_logging_instance.get_security_events.return_value = [
            {'timestamp': '2024-01-01T00:00:00', 'type': 'failed_login'}
        ]
        mock_logging.return_value = mock_logging_instance
        
        # Mock de BackupService
        mock_backup_instance = MagicMock()
        mock_backup_instance.get_backup_info.return_value = [{
            'name': 'backup.tar.gz',
            'size': 1024,
            'created_at': datetime.utcnow().isoformat()
        }]
        mock_backup.return_value = mock_backup_instance
        
        # Mock de MonitoringService
        mock_monitoring_instance = MagicMock()
        mock_monitoring_instance.get_performance_metrics.return_value = {
            'cpu': 50,
            'memory': 70
        }
        mock_monitoring_instance.get_alerts.return_value = [{
            'severity': 'high',
            'message': 'Test alert'
        }]
        mock_monitoring.return_value = mock_monitoring_instance
        
        yield {
            'logging': mock_logging_instance,
            'backup': mock_backup_instance,
            'monitoring': mock_monitoring_instance
        }

def test_get_system_status(mock_services):
    """Prueba la obtención del estado del sistema"""
    dashboard = DashboardService()
    status = dashboard.get_system_status()
    
    assert status['status'] == 'healthy'
    assert 'last_check' in status
    assert 'metrics' in status
    assert 'alerts' in status
    assert 'backups' in status

def test_get_system_metrics(mock_services):
    """Prueba la obtención de métricas del sistema"""
    dashboard = DashboardService()
    metrics = dashboard.get_system_metrics()
    
    assert metrics['logs']['total'] == 1000
    assert metrics['logs']['errors'] == 50
    assert metrics['logs']['warnings'] == 100
    assert metrics['security']['failed_logins'] == 10
    assert metrics['security']['suspicious_activity'] == 5
    assert metrics['performance']['cpu'] == 50
    assert metrics['performance']['memory'] == 70

def test_get_recent_alerts(mock_services):
    """Prueba la obtención de alertas recientes"""
    dashboard = DashboardService()
    alerts = dashboard.get_recent_alerts(hours=24)
    
    assert len(alerts) == 1
    assert alerts[0]['severity'] == 'high'
    assert alerts[0]['message'] == 'Test alert'

def test_get_backup_status(mock_services):
    """Prueba la obtención del estado de backups"""
    dashboard = DashboardService()
    status = dashboard.get_backup_status()
    
    assert status['status'] == 'ok'
    assert status['total_backups'] == 1
    assert status['backup_size'] == 1024
    assert status['last_backup'] is not None

def test_get_log_summary(mock_services):
    """Prueba la obtención del resumen de logs"""
    dashboard = DashboardService()
    summary = dashboard.get_log_summary(days=7)
    
    assert len(summary) == 1
    assert summary[0]['date'] == '2024-01-01'
    assert summary[0]['total'] == 100
    assert summary[0]['errors'] == 5

def test_get_security_events(mock_services):
    """Prueba la obtención de eventos de seguridad"""
    dashboard = DashboardService()
    events = dashboard.get_security_events(days=7)
    
    assert len(events) == 1
    assert events[0]['type'] == 'failed_login'
    assert events[0]['timestamp'] == '2024-01-01T00:00:00'

def test_error_handling(mock_services):
    """Prueba el manejo de errores"""
    dashboard = DashboardService()
    
    # Simular error en LoggingService
    mock_services['logging'].get_log_count.side_effect = Exception('Test error')
    metrics = dashboard.get_system_metrics()
    assert metrics == {}
    
    # Simular error en BackupService
    mock_services['backup'].get_backup_info.side_effect = Exception('Test error')
    status = dashboard.get_backup_status()
    assert status['status'] == 'error'
    assert 'error' in status
    
    # Simular error en MonitoringService
    mock_services['monitoring'].get_alerts.side_effect = Exception('Test error')
    alerts = dashboard.get_recent_alerts()
    assert alerts == []
``` 