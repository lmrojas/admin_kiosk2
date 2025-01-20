# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md

import os

def create_test_directories():
    """Crear directorios para tests"""
    base_dir = 'tests'
    subdirs = ['unit', 'integration', 'e2e']
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        
        # Crear __init__.py para que Python trate los directorios como paquetes
        init_file = os.path.join(path, '__init__.py')
        with open(init_file, 'w') as f:
            f.write('# Directorio de tests\n')
        
        print(f"✅ Directorio {path} creado")

if __name__ == '__main__':
    create_test_directories() 