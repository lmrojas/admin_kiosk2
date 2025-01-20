# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import sys
import subprocess
import platform
import venv
import argparse

class ProjectSetup:
    """
    Clase para configuración y instalación del proyecto de administración de kiosks
    """

    @staticmethod
    def create_virtual_environment(venv_path='venv'):
        """
        Crear entorno virtual de Python
        
        Args:
            venv_path (str): Ruta del entorno virtual
        """
        try:
            print(f"🔧 Creando entorno virtual en {venv_path}...")
            venv.create(venv_path, with_pip=True)
            print("✅ Entorno virtual creado exitosamente")
        except Exception as e:
            print(f"❌ Error creando entorno virtual: {e}")
            sys.exit(1)

    @staticmethod
    def install_dependencies(venv_path='venv'):
        """
        Instalar dependencias del proyecto
        
        Args:
            venv_path (str): Ruta del entorno virtual
        """
        try:
            pip_path = os.path.join(venv_path, 'Scripts' if platform.system() == 'Windows' else 'bin', 'pip')
            
            print("🔍 Instalando dependencias...")
            subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])
            
            # Dependencias adicionales de desarrollo
            dev_dependencies = [
                'pytest', 
                'pytest-cov', 
                'flask-testing', 
                'coverage'
            ]
            subprocess.check_call([pip_path, 'install'] + dev_dependencies)
            
            print("✅ Dependencias instaladas exitosamente")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando dependencias: {e}")
            sys.exit(1)

    @staticmethod
    def initialize_database():
        """
        Inicializar base de datos y migraciones
        """
        try:
            print("🗄️ Inicializando base de datos...")
            subprocess.check_call(['flask', 'db', 'init'])
            subprocess.check_call(['flask', 'db', 'migrate', '-m', 'Initial migration'])
            subprocess.check_call(['flask', 'db', 'upgrade'])
            print("✅ Base de datos inicializada")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error inicializando base de datos: {e}")
            sys.exit(1)

    @staticmethod
    def run_tests():
        """
        Ejecutar suite de pruebas
        """
        try:
            print("🧪 Ejecutando pruebas...")
            subprocess.check_call(['pytest', '-v', '--cov=app'])
            print("✅ Pruebas ejecutadas exitosamente")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error en pruebas: {e}")
            sys.exit(1)

def main():
    """
    Función principal para configuración del proyecto
    """
    parser = argparse.ArgumentParser(description='Configuración del proyecto Admin Kiosk')
    parser.add_argument('--skip-venv', action='store_true', help='Omitir creación de entorno virtual')
    parser.add_argument('--skip-deps', action='store_true', help='Omitir instalación de dependencias')
    parser.add_argument('--skip-db', action='store_true', help='Omitir inicialización de base de datos')
    parser.add_argument('--skip-tests', action='store_true', help='Omitir ejecución de pruebas')
    
    args = parser.parse_args()
    
    print("🚀 Iniciando configuración del proyecto Admin Kiosk")
    
    if not args.skip_venv:
        ProjectSetup.create_virtual_environment()
    
    if not args.skip_deps:
        ProjectSetup.install_dependencies()
    
    if not args.skip_db:
        ProjectSetup.initialize_database()
    
    if not args.skip_tests:
        ProjectSetup.run_tests()
    
    print("✨ Configuración completada exitosamente")

if __name__ == '__main__':
    main() 