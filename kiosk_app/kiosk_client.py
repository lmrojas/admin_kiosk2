"""
Cliente WebSocket que representa un kiosk real.
Maneja la comunicación con el sistema central.
"""
import os
import json
import asyncio
import logging
import platform
import socketio
from datetime import datetime
from typing import Optional, Dict, Any
from .kiosk_app import KioskApp

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KioskClient:
    """Cliente que representa un kiosk real"""
    
    def __init__(self, config_file: str, server_url: str = 'http://localhost:5000'):
        self.config_file = config_file
        self.server_url = server_url
        self.sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=5)
        self.kiosk: Optional[KioskApp] = None
        self.running = False
        self.registered = False
        
        # Configurar handlers de eventos
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('command', self.on_command)
        
    async def initialize(self):
        """Inicializa el cliente y sus recursos"""
        try:
            # Cargar configuración
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            # Inicializar kiosk
            self.kiosk = KioskApp(config)
            logger.info(f"Kiosk inicializado: {self.kiosk.name} ({self.kiosk.serial})")
            
            # Conectar al servidor
            await self.connect_with_retry()
            
        except Exception as e:
            logger.error(f"Error inicializando cliente: {str(e)}")
            raise
            
    async def connect_with_retry(self, max_retries: int = 5):
        """Intenta conectar al servidor con reintentos"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                await self.sio.connect(self.server_url, wait_timeout=10)
                logger.info(f"Conectado al servidor: {self.server_url}")
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Error conectando después de {max_retries} intentos: {str(e)}")
                    raise
                logger.warning(f"Intento {retry_count} fallido, reintentando en 2 segundos...")
                await asyncio.sleep(2)
                
    async def cleanup(self):
        """Limpia recursos y cierra conexiones"""
        try:
            self.running = False
            if self.sio.connected:
                await self.sio.disconnect()
            logger.info("Recursos liberados correctamente")
        except Exception as e:
            logger.error(f"Error durante cleanup: {str(e)}")
            
    async def on_connect(self):
        """Manejador de evento de conexión"""
        logger.info("🔌 Conectado al servidor central")
        
        # Registrar el kiosk
        if self.kiosk and not self.registered:
            try:
                response = await self.sio.emit('registration', {
                    'serial': self.kiosk.serial,
                    'name': self.kiosk.name
                }, callback=True)
                
                if response and response.get('status') == 'registered':
                    self.registered = True
                    logger.info("✅ Kiosk registrado exitosamente")
                else:
                    logger.error(f"❌ Error en registro: {response}")
                    await self.cleanup()
                    
            except Exception as e:
                logger.error(f"Error en registro: {str(e)}")
                await self.cleanup()
        
    async def on_disconnect(self):
        """Manejador de evento de desconexión"""
        logger.info("❌ Desconectado del servidor central")
        self.registered = False
        
    async def on_command(self, data: Dict[str, Any]):
        """Procesa comandos recibidos del servidor"""
        try:
            if not self.kiosk:
                logger.error("Kiosk no inicializado")
                return
                
            response = self.kiosk.process_command(data)
            await self.sio.emit('command_response', response)
            logger.info(f"Respuesta enviada para comando {data.get('command_id')}")
            
        except Exception as e:
            logger.error(f"Error procesando comando: {str(e)}")
            
    async def start_telemetry(self):
        """Inicia el ciclo de envío de telemetría"""
        try:
            self.running = True
            while self.running and self.kiosk:
                try:
                    if self.registered:
                        data = self.kiosk.get_telemetry_data()
                        # Formatear datos según lo esperado por el servidor
                        telemetry = {
                            'serial': self.kiosk.serial,
                            'status': {
                                'current': data['status']
                            },
                            'sensors': {
                                'temperature': data['sensors']['temperature']['value'],
                                'humidity': data['sensors']['humidity']['value'],
                                'door_status': 'closed',  # Simulado
                                'printer_status': 'ok',   # Simulado
                                'network_quality': data['sensors']['network_latency']['value'],
                                'voltage': 220,           # Simulado
                                'ventilation': 'normal'   # Simulado
                            }
                        }
                        await self.sio.emit('kiosk_update', telemetry)
                        logger.debug(f"Telemetría enviada: {telemetry}")
                except Exception as e:
                    logger.error(f"Error enviando telemetría: {str(e)}")
                    
                await asyncio.sleep(5)  # Enviar cada 5 segundos
                
        except Exception as e:
            logger.error(f"Error en ciclo de telemetría: {str(e)}")
            self.running = False
            
    def start(self):
        """Punto de entrada principal"""
        try:
            if platform.system() == 'Windows':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                
            async def main():
                await self.initialize()
                await self.start_telemetry()
                await self.cleanup()
                
            asyncio.run(main())
            
        except KeyboardInterrupt:
            logger.info("Cliente detenido por usuario")
        except Exception as e:
            logger.error(f"Error fatal: {str(e)}")
            raise
            
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python kiosk_client.py <archivo_config.json>")
        sys.exit(1)
        
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Error: Archivo de configuración no encontrado: {config_file}")
        sys.exit(1)
        
    client = KioskClient(config_file)
    client.start() 