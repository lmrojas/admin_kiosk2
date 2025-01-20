# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime
from scripts.backup_logs import main, setup_logging, parse_args

@pytest.fixture
def mock_services():
    """Fixture que proporciona mocks de los servicios"""
    with patch('scripts.backup_logs.BackupService') as mock_backup, \
         patch('scripts.backup_logs.NotificationService') as mock_notify:
        
        # Configurar mock de BackupService
        mock_backup_instance = MagicMock()
        mock_backup_instance.create_backup.return_value = '/path/to/backup.tar.gz'
        mock_backup_instance.verify_backup_integrity.return_value = True
        mock_backup_instance.get_backup_info.return_value = [{
            'path': '/path/to/backup.tar.gz',
            'size': 1024,
            'created_at': datetime.utcnow().isoformat()
        }]
        mock_backup.return_value = mock_backup_instance
        
        # Configurar mock de NotificationService
        mock_notify_instance = MagicMock()
        mock_notify_instance.send_alert.return_value = {'email': True}
        mock_notify.return_value = mock_notify_instance
        
        yield mock_backup_instance, mock_notify_instance

def test_successful_backup(mock_services, capsys):
    """Prueba un backup exitoso sin notificaciones"""
    mock_backup, _ = mock_services
    
    with patch('sys.argv', ['backup_logs.py']):
        exit_code = main()
        
        assert exit_code == 0
        assert mock_backup.create_backup.called
        assert mock_backup.verify_backup_integrity.called
        
        # Verificar logs
        captured = capsys.readouterr()
        assert "Iniciando backup de logs" in captured.err
        assert "Proceso de backup completado" in captured.err

def test_backup_with_notifications(mock_services, capsys):
    """Prueba un backup con notificaciones habilitadas"""
    mock_backup, mock_notify = mock_services
    
    with patch('sys.argv', ['backup_logs.py', '--notify']):
        exit_code = main()
        
        assert exit_code == 0
        assert mock_backup.create_backup.called
        assert mock_notify.send_alert.called
        
        # Verificar contenido de la notificación
        alert = mock_notify.send_alert.call_args[0][0]
        assert alert['severity'] == 'info'
        assert alert['channels'] == ['email']
        assert 'Backup de logs creado exitosamente' in alert['message']

def test_backup_with_cleanup(mock_services, capsys):
    """Prueba un backup con limpieza de archivos antiguos"""
    mock_backup, _ = mock_services
    mock_backup.clean_old_backups.return_value = 2
    
    with patch('sys.argv', ['backup_logs.py', '--clean-old', '--days', '15']):
        exit_code = main()
        
        assert exit_code == 0
        assert mock_backup.clean_old_backups.called
        assert mock_backup.clean_old_backups.call_args[1]['days'] == 15
        
        # Verificar logs
        captured = capsys.readouterr()
        assert "Limpiando backups antiguos" in captured.err
        assert "Se eliminaron 2 backups antiguos" in captured.err

def test_backup_failure(mock_services, capsys):
    """Prueba un backup fallido"""
    mock_backup, mock_notify = mock_services
    mock_backup.create_backup.return_value = None
    
    with patch('sys.argv', ['backup_logs.py', '--notify']):
        exit_code = main()
        
        assert exit_code == 1
        assert mock_backup.create_backup.called
        
        # Verificar notificación de error
        alert = mock_notify.send_alert.call_args[0][0]
        assert alert['severity'] == 'high'
        assert alert['channels'] == ['email', 'slack']
        assert 'Error en el backup de logs' in alert['message']

def test_integrity_check_failure(mock_services, capsys):
    """Prueba un fallo en la verificación de integridad"""
    mock_backup, mock_notify = mock_services
    mock_backup.verify_backup_integrity.return_value = False
    
    with patch('sys.argv', ['backup_logs.py', '--notify']):
        exit_code = main()
        
        assert exit_code == 1
        assert mock_backup.verify_backup_integrity.called
        
        # Verificar notificación de error
        alert = mock_notify.send_alert.call_args[0][0]
        assert alert['severity'] == 'high'
        assert 'Error de integridad' in alert['message']

def test_argument_parsing():
    """Prueba el parsing de argumentos"""
    with patch('sys.argv', [
        'backup_logs.py',
        '--env', 'production',
        '--notify',
        '--clean-old',
        '--days', '45'
    ]):
        args = parse_args()
        
        assert args.env == 'production'
        assert args.notify is True
        assert args.clean_old is True
        assert args.days == 45

def test_logging_setup(tmp_path):
    """Prueba la configuración del logging"""
    with patch('logging.FileHandler') as mock_handler:
        logger = setup_logging()
        
        assert logger.name == 'backup_script'
        assert logger.level == logging.INFO
        assert mock_handler.called

def test_environment_handling(mock_services):
    """Prueba el manejo de diferentes entornos"""
    environments = ['development', 'testing', 'production']
    
    for env in environments:
        with patch('sys.argv', ['backup_logs.py', '--env', env]):
            with patch('scripts.backup_logs.create_app') as mock_create_app:
                exit_code = main()
                
                assert exit_code == 0
                mock_create_app.assert_called_with(env)
``` 