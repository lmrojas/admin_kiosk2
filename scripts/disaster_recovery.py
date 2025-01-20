"""
Script para recuperación ante desastres del sistema.

Funcionalidad:
- Ejecuta procedimientos de recuperación
- Restaura backups de base de datos
- Verifica integridad de datos
- Reconstruye índices y cachés
- Sincroniza estados del sistema

Uso:
python scripts/disaster_recovery.py [--mode MODE]

Argumentos:
--mode: Modo de recuperación (full/partial)
--backup-date: Fecha del backup a restaurar
--verify: Solo verificar sin restaurar

Notas:
- Requiere privilegios administrativos
- Documentar cada ejecución
- Seguir procedimientos de emergencia
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import sys
import shutil
import argparse
import subprocess
import json
from datetime import datetime
import logging

class DisasterRecoveryManager:
    """
    Clase para gestionar la recuperación de desastres del sistema de kiosks
    """

    def __init__(self, backup_dir='backups'):
        """
        Inicializar gestor de recuperación de desastres
        
        Args:
            backup_dir (str): Directorio de respaldos
        """
        self.backup_dir = backup_dir
        self.logger = logging.getLogger('admin_kiosk.disaster_recovery')
        
        # Crear directorio de respaldos si no existe
        os.makedirs(backup_dir, exist_ok=True)

    def create_full_backup(self, output_name=None):
        """
        Crear respaldo completo del sistema
        
        Args:
            output_name (str, opcional): Nombre personalizado para el respaldo
        
        Returns:
            str: Ruta del archivo de respaldo
        """
        try:
            # Generar nombre de respaldo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = output_name or f'admin_kiosk_full_backup_{timestamp}'
            backup_path = os.path.join(self.backup_dir, backup_name)

            # Directorios y archivos a respaldar
            backup_items = [
                'app', 
                'config', 
                'migrations', 
                'requirements.txt', 
                'run.py'
            ]

            # Crear directorio de respaldo
            os.makedirs(backup_path, exist_ok=True)

            # Copiar archivos y directorios
            for item in backup_items:
                if os.path.exists(item):
                    dest_path = os.path.join(backup_path, item)
                    if os.path.isdir(item):
                        shutil.copytree(item, dest_path)
                    else:
                        shutil.copy2(item, dest_path)

            # Respaldar base de datos
            db_backup_path = os.path.join(backup_path, 'database_backup.sql')
            self._backup_database(db_backup_path)

            # Crear archivo de metadatos
            metadata = {
                'timestamp': timestamp,
                'backup_type': 'full',
                'items_backed_up': backup_items
            }
            
            with open(os.path.join(backup_path, 'backup_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            # Comprimir respaldo
            shutil.make_archive(backup_path, 'zip', backup_path)
            backup_zip_path = f'{backup_path}.zip'

            self.logger.info(f"Respaldo completo creado: {backup_zip_path}")
            return backup_zip_path

        except Exception as e:
            self.logger.error(f"Error creando respaldo: {e}")
            raise

    def _backup_database(self, output_path):
        """
        Respaldar base de datos
        
        Args:
            output_path (str): Ruta de salida para el respaldo de base de datos
        """
        try:
            # Comando para respaldar base de datos (ajustar según el tipo de base de datos)
            db_backup_command = [
                'flask', 'db', 'export', 
                '--output', output_path
            ]
            
            subprocess.check_call(db_backup_command)
            self.logger.info(f"Respaldo de base de datos creado: {output_path}")
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error en respaldo de base de datos: {e}")
            raise

    def restore_backup(self, backup_path):
        """
        Restaurar respaldo
        
        Args:
            backup_path (str): Ruta del archivo de respaldo
        """
        try:
            # Descomprimir respaldo
            shutil.unpack_archive(backup_path, self.backup_dir, 'zip')
            
            # Restaurar archivos
            backup_dir_name = os.path.splitext(os.path.basename(backup_path))[0]
            backup_full_path = os.path.join(self.backup_dir, backup_dir_name)
            
            restore_items = [
                'app', 
                'config', 
                'migrations', 
                'requirements.txt', 
                'run.py'
            ]
            
            for item in restore_items:
                src_path = os.path.join(backup_full_path, item)
                if os.path.exists(src_path):
                    dest_path = os.path.join('.', item)
                    if os.path.isdir(src_path):
                        shutil.rmtree(dest_path, ignore_errors=True)
                        shutil.copytree(src_path, dest_path)
                    else:
                        shutil.copy2(src_path, dest_path)
            
            # Restaurar base de datos
            db_backup_path = os.path.join(backup_full_path, 'database_backup.sql')
            if os.path.exists(db_backup_path):
                self._restore_database(db_backup_path)
            
            # Reinstalar dependencias
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            
            self.logger.info(f"Respaldo restaurado desde: {backup_path}")
        
        except Exception as e:
            self.logger.error(f"Error restaurando respaldo: {e}")
            raise

    def _restore_database(self, backup_path):
        """
        Restaurar base de datos
        
        Args:
            backup_path (str): Ruta del respaldo de base de datos
        """
        try:
            # Comando para restaurar base de datos (ajustar según el tipo de base de datos)
            db_restore_command = [
                'flask', 'db', 'import', 
                '--input', backup_path
            ]
            
            subprocess.check_call(db_restore_command)
            self.logger.info(f"Base de datos restaurada desde: {backup_path}")
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error restaurando base de datos: {e}")
            raise

    def list_backups(self):
        """
        Listar respaldos disponibles
        
        Returns:
            list: Lista de respaldos
        """
        backups = []
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.zip'):
                filepath = os.path.join(self.backup_dir, filename)
                backups.append({
                    'filename': filename,
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'created_at': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                })
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)

def main():
    """
    Función principal para gestión de recuperación de desastres
    """
    parser = argparse.ArgumentParser(description='Herramienta de Recuperación de Desastres')
    parser.add_argument('action', choices=['backup', 'restore', 'list'], 
                        help='Acción a realizar')
    parser.add_argument('--path', help='Ruta del archivo de respaldo para restauración')
    parser.add_argument('--name', help='Nombre personalizado para el respaldo')
    
    args = parser.parse_args()
    
    recovery_manager = DisasterRecoveryManager()
    
    try:
        if args.action == 'backup':
            backup_path = recovery_manager.create_full_backup(args.name)
            print(f"Respaldo creado: {backup_path}")
        
        elif args.action == 'restore':
            if not args.path:
                print("Error: Se requiere la ruta del respaldo para restaurar")
                sys.exit(1)
            
            recovery_manager.restore_backup(args.path)
            print(f"Respaldo restaurado desde: {args.path}")
        
        elif args.action == 'list':
            backups = recovery_manager.list_backups()
            print(json.dumps(backups, indent=2))
    
    except Exception as e:
        logging.error(f"Error en recuperación de desastres: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 