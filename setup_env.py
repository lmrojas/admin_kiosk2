# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import os
import subprocess
import sys
import venv

def create_virtual_environment():
    """Crear entorno virtual y instalar dependencias"""
    # Crear entorno virtual
    venv_path = os.path.join(os.path.dirname(__file__), 'venv')
    
    print("🚀 Creando entorno virtual...")
    venv.create(venv_path, with_pip=True)

    # Comando para instalar dependencias
    pip_path = os.path.join(venv_path, 'Scripts' if sys.platform == 'win32' else 'bin', 'pip')
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    install_cmd = [pip_path, 'install', '-r', requirements_path]
    
    print("📦 Instalando dependencias...")
    result = subprocess.run(install_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Entorno virtual creado y dependencias instaladas exitosamente!")
        print("\nPara activar el entorno virtual:")
        if sys.platform == 'win32':
            print("  .\venv\Scripts\Activate")
        else:
            print("  source venv/bin/activate")
    else:
        print("❌ Error instalando dependencias:")
        print(result.stderr)

if __name__ == '__main__':
    create_virtual_environment() 