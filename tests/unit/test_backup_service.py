# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
import os
import shutil
import tarfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.services.backup_service import BackupService

@pytest.fixture
def backup_service(app):
    """Fixture que proporciona una instancia del servicio de backup"""
    with app.test_request_context():
        service = BackupService()
        yield service
        # Limpiar directorios después de las pruebas
        shutil.rmtree(service.backup_dir, ignore_errors=True)
        shutil.rmtree(service.log_dir, ignore_errors=True)

@pytest.fixture
def sample_logs(backup_service):
    """Fixture que crea archivos de log de prueba"""
    # Crear archivos de log
    log_files = ['app.log', 'error.log', 'security.log']
    for file in log_files:
        path = os.path.join(backup_service.log_dir, file)
        with open(path, 'w') as f:
            f.write(f"Test log content for {file}")
    
    # Crear archivo rotado
    rotated_path = os.path.join(backup_service.log_dir, 'app.log.1')
    with open(rotated_path, 'w') as f:
        f.write("Rotated log content")
    
    return log_files

def test_create_backup(backup_service, sample_logs):
    """Prueba la creación de backups"""
    # Crear backup
    backup_path = backup_service.create_backup()
    
    # Verificar que el archivo existe
    assert os.path.exists(backup_path)
    assert backup_path.endswith('.tar.gz')
    
    # Verificar contenido
    with tarfile.open(backup_path, 'r:gz') as tar:
        files = tar.getnames()
        assert all(log in files for log in sample_logs)
        assert 'app.log.1' in files  # Archivo rotado

def test_create_backup_without_rotated(backup_service, sample_logs):
    """Prueba la creación de backups sin archivos rotados"""
    # Crear backup sin archivos rotados
    backup_path = backup_service.create_backup(include_rotated=False)
    
    # Verificar contenido
    with tarfile.open(backup_path, 'r:gz') as tar:
        files = tar.getnames()
        assert all(log in files for log in sample_logs)
        assert 'app.log.1' not in files

def test_clean_old_backups(backup_service):
    """Prueba la limpieza de backups antiguos"""
    # Crear backups con diferentes fechas
    old_backup = os.path.join(backup_service.backup_dir, 'logs_backup_old.tar.gz')
    new_backup = os.path.join(backup_service.backup_dir, 'logs_backup_new.tar.gz')
    
    # Crear archivos
    open(old_backup, 'w').close()
    open(new_backup, 'w').close()
    
    # Modificar fecha del backup antiguo
    old_time = datetime.utcnow() - timedelta(days=31)
    os.utime(old_backup, (old_time.timestamp(), old_time.timestamp()))
    
    # Limpiar backups antiguos
    deleted = backup_service.clean_old_backups(days=30)
    
    assert deleted == 1
    assert not os.path.exists(old_backup)
    assert os.path.exists(new_backup)

def test_restore_backup(backup_service, sample_logs):
    """Prueba la restauración de backups"""
    # Crear y restaurar backup
    backup_path = backup_service.create_backup()
    restore_dir = os.path.join(backup_service.log_dir, 'test_restore')
    
    success = backup_service.restore_backup(backup_path, restore_dir)
    
    assert success
    assert os.path.exists(restore_dir)
    assert all(
        os.path.exists(os.path.join(restore_dir, log))
        for log in sample_logs
    )

def test_restore_nonexistent_backup(backup_service):
    """Prueba la restauración de un backup inexistente"""
    success = backup_service.restore_backup('nonexistent.tar.gz')
    assert not success

def test_get_backup_info(backup_service, sample_logs):
    """Prueba la obtención de información de backups"""
    # Crear algunos backups
    backup_service.create_backup()
    backup_service.create_backup()
    
    # Obtener información
    backups = backup_service.get_backup_info()
    
    assert len(backups) == 2
    for backup in backups:
        assert 'name' in backup
        assert 'size' in backup
        assert 'created_at' in backup
        assert 'path' in backup
        assert os.path.exists(backup['path'])

def test_verify_backup_integrity(backup_service, sample_logs):
    """Prueba la verificación de integridad de backups"""
    # Crear backup válido
    backup_path = backup_service.create_backup()
    assert backup_service.verify_backup_integrity(backup_path)
    
    # Crear backup inválido
    invalid_backup = os.path.join(backup_service.backup_dir, 'invalid.tar.gz')
    with open(invalid_backup, 'w') as f:
        f.write('Invalid content')
    
    assert not backup_service.verify_backup_integrity(invalid_backup)

def test_backup_error_handling(backup_service):
    """Prueba el manejo de errores en operaciones de backup"""
    with patch('tarfile.open') as mock_tar:
        # Simular error al crear backup
        mock_tar.side_effect = Exception('Backup error')
        backup_path = backup_service.create_backup()
        assert backup_path is None

def test_backup_naming_convention(backup_service, sample_logs):
    """Prueba la convención de nombres de los backups"""
    backup_path = backup_service.create_backup()
    backup_name = os.path.basename(backup_path)
    
    # Verificar formato: logs_backup_YYYYMMDD_HHMMSS.tar.gz
    assert backup_name.startswith('logs_backup_')
    assert backup_name.endswith('.tar.gz')
    date_part = backup_name[12:-7]  # Extraer parte de fecha/hora
    assert len(date_part) == 15  # YYYYMMDD_HHMMSS
``` 