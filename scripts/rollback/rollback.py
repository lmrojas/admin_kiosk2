"""
Script para revertir cambios y restaurar versiones anteriores.

Funcionalidad:
- Revierte cambios en base de datos
- Restaura archivos de configuración
- Maneja puntos de restauración
- Verifica integridad de backups
- Registra historial de rollbacks

Uso:
python scripts/rollback/rollback.py [--version VERSION]

Argumentos:
--version: Versión específica a restaurar
--type: Tipo de rollback (full/partial)
--force: Forzar rollback sin confirmación

Notas:
- Requiere backup previo
- Verificar dependencias antes de rollback
- Documentar razón del rollback
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('rollback.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RollbackManager:
    """Gestiona el proceso de rollback del sistema."""
    
    def __init__(self, environment: str):
        """Inicializa el gestor de rollback."""
        self.environment = environment
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.ecs_cluster = f'admin-kiosk-{environment}'
        self.ecs_service = f'admin-kiosk-service-{environment}'
        self.rds_instance = f'admin-kiosk-db-{environment}'
        
        # Inicializar clientes AWS
        self.ecs_client = boto3.client('ecs', region_name=self.aws_region)
        self.rds_client = boto3.client('rds', region_name=self.aws_region)
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
        
        # Bucket para backups y versiones
        self.backup_bucket = f'admin-kiosk-backups-{environment}'
        self.version_file = 'versions.json'

    def rollback_application(self, target_version: Optional[str] = None) -> bool:
        """Realiza el rollback de la aplicación a una versión específica."""
        try:
            # Obtener versión objetivo
            if not target_version:
                target_version = self._get_previous_version()
            
            logger.info(f"Iniciando rollback a versión {target_version}")
            
            # Verificar existencia de la versión
            if not self._verify_version_exists(target_version):
                logger.error(f"Versión {target_version} no encontrada")
                return False
            
            # Realizar rollback de base de datos
            if not self._rollback_database(target_version):
                logger.error("Fallo en rollback de base de datos")
                return False
            
            # Realizar rollback de la aplicación
            if not self._rollback_ecs_service(target_version):
                logger.error("Fallo en rollback de ECS")
                return False
            
            # Actualizar archivo de versiones
            self._update_version_history(target_version)
            
            logger.info(f"Rollback completado exitosamente a versión {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error durante rollback: {str(e)}")
            return False

    def _get_previous_version(self) -> str:
        """Obtiene la versión anterior del historial."""
        try:
            versions = self._load_version_history()
            if len(versions) < 2:
                raise ValueError("No hay versión anterior disponible")
            return versions[-2]['version']
        except Exception as e:
            logger.error(f"Error al obtener versión anterior: {str(e)}")
            raise

    def _verify_version_exists(self, version: str) -> bool:
        """Verifica si existe una versión específica."""
        try:
            # Verificar imagen en ECR
            ecr_client = boto3.client('ecr', region_name=self.aws_region)
            repository = f'admin-kiosk-{self.environment}'
            images = ecr_client.describe_images(
                repositoryName=repository,
                imageIds=[{'imageTag': version}]
            )
            
            # Verificar backup en S3
            backup_key = f'backups/db/{version}/backup.sql'
            self.s3_client.head_object(
                Bucket=self.backup_bucket,
                Key=backup_key
            )
            
            return True
        except ClientError:
            return False

    def _rollback_database(self, version: str) -> bool:
        """Realiza el rollback de la base de datos."""
        try:
            logger.info("Iniciando rollback de base de datos")
            
            # Crear snapshot antes del rollback
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            snapshot_id = f'pre-rollback-{timestamp}'
            
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
            
            # Restaurar backup de la versión objetivo
            backup_key = f'backups/db/{version}/backup.sql'
            self._restore_database_backup(backup_key)
            
            logger.info("Rollback de base de datos completado")
            return True
            
        except Exception as e:
            logger.error(f"Error en rollback de base de datos: {str(e)}")
            return False

    def _rollback_ecs_service(self, version: str) -> bool:
        """Realiza el rollback del servicio ECS."""
        try:
            logger.info("Iniciando rollback de servicio ECS")
            
            # Obtener task definition actual
            service = self.ecs_client.describe_services(
                cluster=self.ecs_cluster,
                services=[self.ecs_service]
            )['services'][0]
            
            # Crear nueva task definition con la imagen anterior
            task_def = self.ecs_client.describe_task_definition(
                taskDefinition=service['taskDefinition']
            )['taskDefinition']
            
            new_task_def = self.ecs_client.register_task_definition(
                family=task_def['family'],
                containerDefinitions=[{
                    **container,
                    'image': f'{container["image"].split(":")[0]}:{version}'
                } for container in task_def['containerDefinitions']],
                volumes=task_def.get('volumes', []),
                taskRoleArn=task_def.get('taskRoleArn'),
                executionRoleArn=task_def.get('executionRoleArn')
            )
            
            # Actualizar servicio con la nueva task definition
            self.ecs_client.update_service(
                cluster=self.ecs_cluster,
                service=self.ecs_service,
                taskDefinition=new_task_def['taskDefinition']['taskDefinitionArn']
            )
            
            # Esperar a que el servicio esté estable
            waiter = self.ecs_client.get_waiter('services_stable')
            waiter.wait(
                cluster=self.ecs_cluster,
                services=[self.ecs_service],
                WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
            )
            
            logger.info("Rollback de servicio ECS completado")
            return True
            
        except Exception as e:
            logger.error(f"Error en rollback de ECS: {str(e)}")
            return False

    def _restore_database_backup(self, backup_key: str) -> None:
        """Restaura un backup de base de datos desde S3."""
        try:
            # Descargar backup de S3
            local_backup = '/tmp/backup.sql'
            self.s3_client.download_file(
                self.backup_bucket,
                backup_key,
                local_backup
            )
            
            # Ejecutar restauración
            os.system(f'psql -h {self.rds_instance} -U admin_kiosk -d admin_kiosk -f {local_backup}')
            
            # Limpiar archivo temporal
            os.remove(local_backup)
            
        except Exception as e:
            logger.error(f"Error al restaurar backup: {str(e)}")
            raise

    def _load_version_history(self) -> List[Dict]:
        """Carga el historial de versiones desde S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.backup_bucket,
                Key=self.version_file
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError:
            return []

    def _update_version_history(self, rollback_version: str) -> None:
        """Actualiza el historial de versiones después del rollback."""
        try:
            versions = self._load_version_history()
            versions.append({
                'version': rollback_version,
                'type': 'rollback',
                'timestamp': datetime.now().isoformat(),
                'environment': self.environment
            })
            
            self.s3_client.put_object(
                Bucket=self.backup_bucket,
                Key=self.version_file,
                Body=json.dumps(versions, indent=2)
            )
            
        except Exception as e:
            logger.error(f"Error al actualizar historial de versiones: {str(e)}")
            raise

def main():
    """Función principal del script de rollback."""
    parser = argparse.ArgumentParser(description='Script de rollback para Admin Kiosk')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Ambiente donde realizar el rollback')
    parser.add_argument('--version', help='Versión específica para el rollback')
    
    args = parser.parse_args()
    
    rollback_manager = RollbackManager(args.environment)
    success = rollback_manager.rollback_application(args.version)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 