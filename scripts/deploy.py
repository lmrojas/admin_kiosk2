"""
Script para automatizar el despliegue del sistema.

Funcionalidad:
- Automatiza proceso de despliegue
- Verifica dependencias y requisitos
- Ejecuta migraciones de base de datos
- Actualiza configuraciones
- Reinicia servicios necesarios

Uso:
python scripts/deploy.py [--env ENV]

Argumentos:
--env: Ambiente de despliegue (prod/staging)
--rollback: Revertir último despliegue
--dry-run: Simular sin aplicar cambios

Notas:
- Requiere configuración de ambiente
- Hacer backup antes de desplegar
- Verificar logs post-despliegue
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import sys
import subprocess
import platform
import argparse
import shutil
import json
from datetime import datetime

class Deployer:
    """
    Clase para gestionar el despliegue del proyecto de administración de kiosks
    """

    @staticmethod
    def _get_deployment_config():
        """
        Obtener configuración de despliegue
        
        Returns:
            dict: Configuración de despliegue
        """
        config_path = os.path.join('config', 'deployment.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'production': {
                    'host': 'example.com',
                    'user': 'admin',
                    'path': '/var/www/admin_kiosk'
                }
            }

    @staticmethod
    def prepare_deployment_package(environment='production'):
        """
        Preparar paquete de despliegue
        
        Args:
            environment (str): Entorno de despliegue
        
        Returns:
            str: Ruta del paquete de despliegue
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deploy_dir = f'deploy_{timestamp}'
        
        try:
            # Crear directorio de despliegue
            os.makedirs(deploy_dir, exist_ok=True)
            
            # Copiar archivos necesarios
            files_to_copy = [
                'app', 
                'config', 
                'requirements.txt', 
                'run.py', 
                'migrations'
            ]
            
            for item in files_to_copy:
                if os.path.exists(item):
                    if os.path.isdir(item):
                        shutil.copytree(item, os.path.join(deploy_dir, item))
                    else:
                        shutil.copy2(item, os.path.join(deploy_dir, item))
            
            # Crear archivo de configuración de entorno
            env_config = {
                'FLASK_ENV': environment,
                'SECRET_KEY': os.urandom(24).hex(),
                'DATABASE_URL': os.environ.get('PRODUCTION_DATABASE_URL', '')
            }
            
            with open(os.path.join(deploy_dir, '.env'), 'w') as f:
                for key, value in env_config.items():
                    f.write(f"{key}={value}\n")
            
            # Crear archivo de requerimientos específicos del entorno
            subprocess.check_call([sys.executable, '-m', 'pip', 'freeze'], 
                                  stdout=open(os.path.join(deploy_dir, 'requirements_deployed.txt'), 'w'))
            
            # Comprimir paquete de despliegue
            shutil.make_archive(deploy_dir, 'zip', deploy_dir)
            
            print(f"✅ Paquete de despliegue creado: {deploy_dir}.zip")
            return f"{deploy_dir}.zip"
        
        except Exception as e:
            print(f"❌ Error preparando paquete de despliegue: {e}")
            sys.exit(1)

    @staticmethod
    def deploy_to_server(package_path, environment='production'):
        """
        Desplegar paquete a servidor remoto
        
        Args:
            package_path (str): Ruta del paquete de despliegue
            environment (str): Entorno de despliegue
        """
        try:
            # Obtener configuración de despliegue
            deploy_config = Deployer._get_deployment_config()[environment]
            
            # Comandos de despliegue
            ssh_commands = [
                f'mkdir -p {deploy_config["path"]}',
                f'cd {deploy_config["path"]}',
                f'unzip -o {package_path}',
                'python3 -m venv venv',
                'source venv/bin/activate',
                'pip install -r requirements_deployed.txt',
                'flask db upgrade',
                'sudo systemctl restart admin_kiosk'
            ]
            
            # Comando de SCP para copiar paquete
            scp_command = [
                'scp', 
                package_path, 
                f'{deploy_config["user"]}@{deploy_config["host"]}:{deploy_config["path"]}'
            ]
            
            # Comando SSH para ejecutar comandos de despliegue
            ssh_command = [
                'ssh', 
                f'{deploy_config["user"]}@{deploy_config["host"]}',
                ' && '.join(ssh_commands)
            ]
            
            # Ejecutar despliegue
            subprocess.check_call(scp_command)
            subprocess.check_call(ssh_command)
            
            print(f"✅ Despliegue en {environment} completado exitosamente")
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Error en despliegue: {e}")
            sys.exit(1)

    @staticmethod
    def rollback(environment='production'):
        """
        Realizar rollback a la versión anterior
        
        Args:
            environment (str): Entorno de despliegue
        """
        try:
            deploy_config = Deployer._get_deployment_config()[environment]
            
            rollback_commands = [
                f'cd {deploy_config["path"]}',
                'ls -t *.zip | tail -n 2 | head -n 1',  # Obtener segundo paquete más reciente
                'unzip -o $(ls -t *.zip | tail -n 2 | head -n 1)',
                'source venv/bin/activate',
                'pip install -r requirements_deployed.txt',
                'flask db upgrade',
                'sudo systemctl restart admin_kiosk'
            ]
            
            ssh_command = [
                'ssh', 
                f'{deploy_config["user"]}@{deploy_config["host"]}',
                ' && '.join(rollback_commands)
            ]
            
            subprocess.check_call(ssh_command)
            
            print(f"✅ Rollback en {environment} completado exitosamente")
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Error en rollback: {e}")
            sys.exit(1)

def main():
    """
    Función principal para gestión de despliegue
    """
    parser = argparse.ArgumentParser(description='Herramienta de despliegue para Admin Kiosk')
    parser.add_argument('action', choices=['prepare', 'deploy', 'rollback'], 
                        help='Acción a realizar')
    parser.add_argument('--env', default='production', 
                        help='Entorno de despliegue')
    
    args = parser.parse_args()
    
    print(f"🚀 Iniciando {args.action} en entorno {args.env}")
    
    if args.action == 'prepare':
        Deployer.prepare_deployment_package(args.env)
    elif args.action == 'deploy':
        # Preparar paquete y desplegarlo
        package = Deployer.prepare_deployment_package(args.env)
        Deployer.deploy_to_server(package, args.env)
    elif args.action == 'rollback':
        Deployer.rollback(args.env)
    
    print("✨ Proceso completado")

if __name__ == '__main__':
    main() 