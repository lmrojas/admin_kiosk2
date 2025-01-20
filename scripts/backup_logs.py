# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Agregar directorio raíz al path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from app import create_app
from app.services.backup_service import BackupService
from app.services.notification_service import NotificationService

def setup_logging():
    """Configura el logging para el script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backup_logs.log')
        ]
    )
    return logging.getLogger('backup_script')

def parse_args():
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Script de backup de logs')
    parser.add_argument(
        '--env',
        default='production',
        choices=['development', 'testing', 'production'],
        help='Entorno de ejecución'
    )
    parser.add_argument(
        '--notify',
        action='store_true',
        help='Enviar notificaciones de resultado'
    )
    parser.add_argument(
        '--clean-old',
        action='store_true',
        help='Limpiar backups antiguos'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Días de antigüedad para limpiar backups'
    )
    return parser.parse_args()

def main():
    """Función principal del script"""
    args = parse_args()
    logger = setup_logging()
    
    try:
        # Crear aplicación Flask
        app = create_app(args.env)
        
        with app.app_context():
            backup_service = BackupService()
            notification_service = NotificationService() if args.notify else None
            
            # Crear backup
            logger.info("Iniciando backup de logs...")
            backup_path = backup_service.create_backup()
            
            if not backup_path:
                raise Exception("Error creando backup")
            
            # Verificar integridad
            if not backup_service.verify_backup_integrity(backup_path):
                raise Exception("Error de integridad en el backup")
            
            logger.info(f"Backup creado exitosamente: {backup_path}")
            
            # Limpiar backups antiguos si se especificó
            if args.clean_old:
                logger.info(f"Limpiando backups antiguos (>{args.days} días)...")
                deleted = backup_service.clean_old_backups(days=args.days)
                logger.info(f"Se eliminaron {deleted} backups antiguos")
            
            # Obtener información del backup
            backup_info = next(
                (b for b in backup_service.get_backup_info() 
                 if b['path'] == backup_path),
                {}
            )
            
            # Enviar notificación si se solicitó
            if args.notify and notification_service:
                alert = {
                    'name': 'backup_logs',
                    'severity': 'info',
                    'message': (
                        f"Backup de logs creado exitosamente\n"
                        f"Archivo: {os.path.basename(backup_path)}\n"
                        f"Tamaño: {backup_info.get('size', 0)} bytes"
                    ),
                    'timestamp': datetime.utcnow().isoformat(),
                    'channels': ['email']
                }
                
                notification_service.send_alert(alert)
            
            logger.info("Proceso de backup completado")
            return 0
            
    except Exception as e:
        logger.error(f"Error en el proceso de backup: {str(e)}")
        
        # Enviar notificación de error si se solicitó
        if args.notify and notification_service:
            alert = {
                'name': 'backup_logs_error',
                'severity': 'high',
                'message': f"Error en el backup de logs: {str(e)}",
                'timestamp': datetime.utcnow().isoformat(),
                'channels': ['email', 'slack']
            }
            
            notification_service.send_alert(alert)
        
        return 1

if __name__ == '__main__':
    sys.exit(main()) 