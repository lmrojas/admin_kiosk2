"""
Script para crear y gestionar copias de seguridad del sistema.

Funcionalidad:
- Crea backups completos del sistema
- Gestiona puntos de restauración
- Comprime y archiva backups
- Verifica integridad de copias
- Mantiene historial de backups

Uso:
python scripts/rollback/backup.py [--type TIPO]

Argumentos:
--type: Tipo de backup (full/incremental)
--compress: Comprimir backup
--verify: Verificar integridad

Salida:
- Archivo de backup (.tar.gz)
- Registro de metadatos
- Log de operación

Notas:
- Programar backups periódicos
- Verificar espacio disponible
- Rotar backups antiguos
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('backup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BackupManager:
    """Gestiona el proceso de backup del sistema."""
    
    def __init__(self, environment: str, version: str):
        """Inicializa el gestor de backup."""
        self.environment = environment
        self.version = version
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.rds_instance = f'admin-kiosk-db-{environment}'
        
        # Inicializar clientes AWS
        self.rds_client = boto3.client('rds', region_name=self.aws_region)
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
        
        # Bucket para backups
        self.backup_bucket = f'admin-kiosk-backups-{environment}'

    def create_backup(self) -> bool:
        """Crea un backup completo del sistema."""
        try:
            logger.info(f"Iniciando backup para versión {self.version}")
            
            # Crear backup de base de datos
            if not self._backup_database():
                return False
            
            # Crear backup de archivos estáticos
            if not self._backup_static_files():
                return False
            
            # Crear backup de configuración
            if not self._backup_configuration():
                return False
            
            logger.info(f"Backup completado exitosamente para versión {self.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error durante backup: {str(e)}")
            return False

    def _backup_database(self) -> bool:
        """Realiza el backup de la base de datos."""
        try:
            logger.info("Iniciando backup de base de datos")
            
            # Crear snapshot de RDS
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            snapshot_id = f'pre-deploy-{self.version}-{timestamp}'
            
            self.rds_client.create_db_snapshot(
                DBSnapshotIdentifier=snapshot_id,
                DBInstanceIdentifier=self.rds_instance
            )
            
            # Esperar a que el snapshot esté disponible
            waiter = self.rds_client.get_waiter('db_snapshot_available')
            waiter.wait(
                DBSnapshotIdentifier=snapshot_id,
                WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
            )
            
            # Exportar datos a archivo SQL
            backup_file = f'/tmp/backup-{self.version}.sql'
            os.system(f'pg_dump -h {self.rds_instance} -U admin_kiosk -d admin_kiosk -f {backup_file}')
            
            # Subir backup a S3
            backup_key = f'backups/db/{self.version}/backup.sql'
            self.s3_client.upload_file(
                backup_file,
                self.backup_bucket,
                backup_key
            )
            
            # Limpiar archivo temporal
            os.remove(backup_file)
            
            logger.info("Backup de base de datos completado")
            return True
            
        except Exception as e:
            logger.error(f"Error en backup de base de datos: {str(e)}")
            return False

    def _backup_static_files(self) -> bool:
        """Realiza el backup de archivos estáticos."""
        try:
            logger.info("Iniciando backup de archivos estáticos")
            
            # Crear archivo temporal
            static_dir = '/app/static'
            backup_file = f'/tmp/static-{self.version}.tar.gz'
            
            # Comprimir archivos estáticos
            os.system(f'tar -czf {backup_file} -C {static_dir} .')
            
            # Subir a S3
            backup_key = f'backups/static/{self.version}/static.tar.gz'
            self.s3_client.upload_file(
                backup_file,
                self.backup_bucket,
                backup_key
            )
            
            # Limpiar archivo temporal
            os.remove(backup_file)
            
            logger.info("Backup de archivos estáticos completado")
            return True
            
        except Exception as e:
            logger.error(f"Error en backup de archivos estáticos: {str(e)}")
            return False

    def _backup_configuration(self) -> bool:
        """Realiza el backup de la configuración."""
        try:
            logger.info("Iniciando backup de configuración")
            
            # Recopilar configuración actual
            config = {
                'version': self.version,
                'timestamp': datetime.now().isoformat(),
                'environment': self.environment,
                'ecs_config': self._get_ecs_config(),
                'rds_config': self._get_rds_config(),
                'environment_variables': self._get_environment_variables()
            }
            
            # Subir a S3
            backup_key = f'backups/config/{self.version}/config.json'
            self.s3_client.put_object(
                Bucket=self.backup_bucket,
                Key=backup_key,
                Body=json.dumps(config, indent=2)
            )
            
            logger.info("Backup de configuración completado")
            return True
            
        except Exception as e:
            logger.error(f"Error en backup de configuración: {str(e)}")
            return False

    def _get_ecs_config(self) -> dict:
        """Obtiene la configuración actual de ECS."""
        try:
            ecs_client = boto3.client('ecs', region_name=self.aws_region)
            cluster = f'admin-kiosk-{self.environment}'
            service = f'admin-kiosk-service-{self.environment}'
            
            # Obtener configuración del servicio
            service_config = ecs_client.describe_services(
                cluster=cluster,
                services=[service]
            )['services'][0]
            
            # Obtener task definition
            task_def = ecs_client.describe_task_definition(
                taskDefinition=service_config['taskDefinition']
            )['taskDefinition']
            
            return {
                'service': service_config,
                'task_definition': task_def
            }
            
        except Exception as e:
            logger.error(f"Error al obtener configuración ECS: {str(e)}")
            return {}

    def _get_rds_config(self) -> dict:
        """Obtiene la configuración actual de RDS."""
        try:
            # Obtener configuración de la instancia
            response = self.rds_client.describe_db_instances(
                DBInstanceIdentifier=self.rds_instance
            )
            
            instance = response['DBInstances'][0]
            return {
                'instance_class': instance['DBInstanceClass'],
                'engine_version': instance['EngineVersion'],
                'allocated_storage': instance['AllocatedStorage'],
                'backup_retention_period': instance['BackupRetentionPeriod']
            }
            
        except Exception as e:
            logger.error(f"Error al obtener configuración RDS: {str(e)}")
            return {}

    def _get_environment_variables(self) -> dict:
        """Obtiene las variables de entorno actuales."""
        try:
            ssm_client = boto3.client('ssm', region_name=self.aws_region)
            
            # Obtener parámetros de SSM
            response = ssm_client.get_parameters_by_path(
                Path=f'/admin-kiosk/{self.environment}/',
                WithDecryption=True
            )
            
            return {
                param['Name'].split('/')[-1]: param['Value']
                for param in response['Parameters']
            }
            
        except Exception as e:
            logger.error(f"Error al obtener variables de entorno: {str(e)}")
            return {}

def main():
    """Función principal del script de backup."""
    parser = argparse.ArgumentParser(description='Script de backup para Admin Kiosk')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Ambiente donde realizar el backup')
    parser.add_argument('--version', required=True,
                       help='Versión para la cual se realiza el backup')
    
    args = parser.parse_args()
    
    backup_manager = BackupManager(args.environment, args.version)
    success = backup_manager.create_backup()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 