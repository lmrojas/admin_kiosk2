# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import shutil
import logging
import tarfile
from datetime import datetime, timedelta
from typing import List, Optional
from flask import current_app
from pathlib import Path

class BackupService:
    """Servicio para gestionar backups de logs"""
    
    def __init__(self):
        """Inicializa el servicio de backup"""
        self.logger = logging.getLogger('backup')
        self.log_dir = os.path.join(current_app.root_path, 'logs')
        self.backup_dir = os.path.join(current_app.root_path, 'backups', 'logs')
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Asegura que existan los directorios necesarios"""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self, include_rotated: bool = True) -> Optional[str]:
        """
        Crea un backup de los archivos de log
        
        Args:
            include_rotated: Si se incluyen los archivos rotados
            
        Returns:
            str: Ruta del archivo de backup creado o None si falla
        """
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_name = f'logs_backup_{timestamp}.tar.gz'
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            # Crear archivo tar.gz
            with tarfile.open(backup_path, 'w:gz') as tar:
                for log_file in self._get_log_files(include_rotated):
                    arcname = os.path.basename(log_file)
                    tar.add(log_file, arcname=arcname)
            
            self.logger.info(f"Backup creado exitosamente: {backup_name}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creando backup: {str(e)}")
            return None
    
    def _get_log_files(self, include_rotated: bool) -> List[str]:
        """
        Obtiene la lista de archivos de log a respaldar
        
        Args:
            include_rotated: Si se incluyen los archivos rotados
            
        Returns:
            List[str]: Lista de rutas de archivos
        """
        log_files = []
        for file in os.listdir(self.log_dir):
            if file.endswith('.log'):
                log_files.append(os.path.join(self.log_dir, file))
            elif include_rotated and file.endswith('.log.1'):
                log_files.append(os.path.join(self.log_dir, file))
        return log_files
    
    def clean_old_backups(self, days: int = 30) -> int:
        """
        Elimina backups antiguos
        
        Args:
            days: Días de antigüedad para eliminar
            
        Returns:
            int: Número de archivos eliminados
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            deleted_count = 0
            
            for backup_file in os.listdir(self.backup_dir):
                if not backup_file.startswith('logs_backup_'):
                    continue
                
                backup_path = os.path.join(self.backup_dir, backup_file)
                file_date = datetime.fromtimestamp(os.path.getctime(backup_path))
                
                if file_date < cutoff_date:
                    os.remove(backup_path)
                    deleted_count += 1
                    self.logger.info(f"Backup antiguo eliminado: {backup_file}")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error limpiando backups antiguos: {str(e)}")
            return 0
    
    def restore_backup(self, backup_path: str, target_dir: Optional[str] = None) -> bool:
        """
        Restaura un backup de logs
        
        Args:
            backup_path: Ruta del archivo de backup
            target_dir: Directorio destino (opcional)
            
        Returns:
            bool: True si la restauración fue exitosa
        """
        try:
            if not os.path.exists(backup_path):
                self.logger.error(f"Archivo de backup no encontrado: {backup_path}")
                return False
            
            restore_dir = target_dir or os.path.join(self.log_dir, 'restored')
            os.makedirs(restore_dir, exist_ok=True)
            
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(path=restore_dir)
            
            self.logger.info(f"Backup restaurado en: {restore_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restaurando backup: {str(e)}")
            return False
    
    def get_backup_info(self) -> List[dict]:
        """
        Obtiene información de los backups existentes
        
        Returns:
            List[dict]: Lista con información de cada backup
        """
        try:
            backups = []
            for backup_file in os.listdir(self.backup_dir):
                if not backup_file.startswith('logs_backup_'):
                    continue
                
                backup_path = os.path.join(self.backup_dir, backup_file)
                file_stats = os.stat(backup_path)
                
                backups.append({
                    'name': backup_file,
                    'size': file_stats.st_size,
                    'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    'path': backup_path
                })
            
            return sorted(backups, key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error obteniendo información de backups: {str(e)}")
            return []
    
    def verify_backup_integrity(self, backup_path: str) -> bool:
        """
        Verifica la integridad de un archivo de backup
        
        Args:
            backup_path: Ruta del archivo de backup
            
        Returns:
            bool: True si el backup está íntegro
        """
        try:
            with tarfile.open(backup_path, 'r:gz') as tar:
                # Verificar que el archivo sea un tar válido
                tar.getmembers()
                
                # Verificar que contenga al menos un archivo de log
                has_logs = any(
                    member.name.endswith('.log') 
                    for member in tar.getmembers()
                )
                
                if not has_logs:
                    self.logger.warning(
                        f"Backup no contiene archivos de log: {backup_path}"
                    )
                    return False
                
            return True
            
        except Exception as e:
            self.logger.error(
                f"Error verificando integridad del backup {backup_path}: {str(e)}"
            )
            return False
``` 