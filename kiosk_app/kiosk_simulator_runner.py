"""
Runner para simular múltiples kiosks conectándose al servidor central.
"""

import logging
import socketio
import time
from typing import Dict
from kiosk_app import Kiosk, KioskConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KioskSimulatorRunner:
    """Maneja múltiples kiosks y su comunicación con el servidor central"""
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """Inicializa el runner con la URL del servidor"""
        self.server_url = server_url
        self.kiosks: Dict[str, Kiosk] = {}
        self.sio = socketio.Client(logger=True, engineio_logger=True)
        self._setup_socket_handlers()
        
    def _setup_socket_handlers(self):
        """Configura los manejadores de eventos del socket"""
        @self.sio.event
        def connect():
            logger.info("Conectado al servidor central")
            
        @self.sio.event
        def disconnect():
            logger.info("Desconectado del servidor central")
            
        @self.sio.on('command')
        def on_command(data):
            """Maneja comandos recibidos del servidor"""
            serial = data.get('serial')
            command = data.get('command')
            params = data.get('params', {})
            
            if serial in self.kiosks:
                kiosk = self.kiosks[serial]
                response = kiosk.process_command(command, params)
                self.sio.emit('command_response', {
                    'serial': serial,
                    'command': command,
                    'response': response
                })
            else:
                logger.error(f"Comando recibido para kiosk desconocido: {serial}")
                
    def add_kiosk(self, serial: str, name: str, location: dict, timezone: str = "UTC"):
        """Añade un nuevo kiosk al simulador"""
        config = KioskConfig(serial=serial, name=name, location=location, timezone=timezone)
        self.kiosks[serial] = Kiosk(config)
        logger.info(f"Kiosk añadido: {serial}")
        
    def register_kiosk(self, serial: str):
        """Registra un kiosk con el servidor central"""
        if serial not in self.kiosks:
            logger.error(f"Kiosk no encontrado: {serial}")
            return False
            
        kiosk = self.kiosks[serial]
        registration_data = kiosk.get_registration_data()
        
        try:
            self.sio.emit('register', registration_data, callback=lambda response: self._handle_registration_response(serial, response))
            logger.info(f"Solicitud de registro enviada para kiosk: {serial}")
            return True
        except Exception as e:
            logger.error(f"Error registrando kiosk {serial}: {e}")
            return False
            
    def _handle_registration_response(self, serial: str, response: dict):
        """Maneja la respuesta del registro de un kiosk"""
        if response.get('status') == 'success':
            self.kiosks[serial].is_registered = True
            self.kiosks[serial].socket_id = response.get('socket_id')
            logger.info(f"Kiosk {serial} registrado exitosamente")
        else:
            logger.error(f"Error registrando kiosk {serial}: {response.get('message')}")
            
    def start(self):
        """Inicia la simulación de todos los kiosks"""
        try:
            # Conectar al servidor
            self.sio.connect(self.server_url)
            
            # Registrar kiosks
            for serial in self.kiosks:
                self.register_kiosk(serial)
                
            # Iniciar envío de telemetría
            while True:
                for serial, kiosk in self.kiosks.items():
                    if kiosk.is_registered:
                        telemetry = kiosk.get_telemetry_data()
                        self.sio.emit('telemetry', {
                            'serial': serial,
                            'data': telemetry
                        })
                time.sleep(5)  # Enviar telemetría cada 5 segundos
                
        except Exception as e:
            logger.error(f"Error en la simulación: {e}")
        finally:
            if self.sio.connected:
                self.sio.disconnect()

if __name__ == "__main__":
    # Crear instancia del runner
    runner = KioskSimulatorRunner()
    
    # Agregar kiosks existentes
    kiosks_data = [
        ("7a43be88-3839-455c-88d4-9e7089147538", "K0001", {"latitude": -34.6037, "longitude": -58.3816, "accuracy": 10.0}, "America/Argentina/Buenos_Aires"),
        ("5ea29aa4-59af-4925-b3fd-38556fd8abf8", "K0002", {"latitude": -34.6037, "longitude": -58.3816, "accuracy": 10.0}, "America/Argentina/Buenos_Aires"),
        ("3c8acd62-2ef1-405b-9957-1925044ad9d8", "k0003", {"latitude": -34.6037, "longitude": -58.3816, "accuracy": 10.0}, "America/Argentina/Buenos_Aires")
    ]
    
    for serial, name, location, timezone in kiosks_data:
        runner.add_kiosk(serial, name, location, timezone)
    
    # Iniciar simulación
    runner.start() 