"""
Script maestro para orquestar la simulación completa de kiosks.
Ejecuta los tres pasos en secuencia:
1. Registro en BD central
2. Creación de kiosks simulados
3. Conexión al sistema central
"""
import os
import sys
import time
import logging
import subprocess
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kiosk_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_system_ready():
    """Verifica que el sistema esté listo para la simulación"""
    try:
        # Detener procesos Python previos
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        time.sleep(2)
        
        # Verificar puerto 5000
        result = subprocess.run(['netstat', '-ano', '|', 'findstr', ':5000'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              shell=True)
        
        if result.stdout:
            raise Exception("Puerto 5000 en uso. Asegúrate que no haya instancias previas.")
            
        return True
        
    except Exception as e:
        logger.error(f"Error preparando el sistema: {str(e)}")
        return False

def main():
    """Función principal de simulación"""
    logger.info("Iniciando simulación de 30 kiosks...")
    
    # 1. Verificar sistema
    logger.info("Verificando estado del sistema central...")
    if not check_system_ready():
        logger.error("Sistema no está listo. Abortando simulación.")
        return
        
    # 2. Iniciar sistema central
    logger.info("Iniciando sistema central...")
    try:
        subprocess.Popen([sys.executable, 'run.py'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        time.sleep(5)  # Esperar que inicie
    except Exception as e:
        logger.error(f"Error iniciando sistema central: {str(e)}")
        return
        
    # 3. Registrar kiosks
    logger.info("Registrando kiosks en sistema central...")
    try:
        subprocess.run([sys.executable, 'register_kiosks.py'],
                      check=True)
        time.sleep(2)
    except Exception as e:
        logger.error(f"Error registrando kiosks: {str(e)}")
        return
        
    # 4. Iniciar simulación
    logger.info("Iniciando simulación de kiosks...")
    try:
        os.chdir('kiosk_app')
        subprocess.Popen([sys.executable, 'kiosk_spawner.py'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        logger.info("Simulación iniciada exitosamente")
    except Exception as e:
        logger.error(f"Error iniciando simulación: {str(e)}")
        return

if __name__ == "__main__":
    main() 