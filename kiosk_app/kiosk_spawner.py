"""
Módulo para gestionar múltiples instancias de kiosks
"""
import socketio
import json
import os
import logging
from .kiosk_app import KioskApp

logger = logging.getLogger(__name__)

class KioskSpawner:
    """Clase que gestiona múltiples instancias de kiosks y su conexión al sistema central"""
    
    def __init__(self, config_dir: str, server_url: str = 'http://localhost:5000'):
        self.config_dir = config_dir
        self.server_url = server_url
        self.running = False
        self.sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=5, reconnection_delay=1)
        self.kiosks = {}
        self.registered_kiosks = {}
        
        # Configurar handlers de eventos
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('command', self.handle_command)
        
    async def initialize(self):
        """Inicializa el spawner y sus recursos"""
        try:
            # Inicializar instancias de kiosks
            self.kiosks = KioskApp.initialize_instances(self.config_dir)
            logger.info(f"Inicializados {len(self.kiosks)} kiosks")
            
            # Intentar conectar con reintentos
            retry_count = 0
            max_retries = 5
            while retry_count < max_retries:
                try:
                    await self.sio.connect(self.server_url, wait_timeout=10)
                    logger.info(f"Conectado al servidor central: {self.server_url}")
                    
                    # Registrar kiosks en el servidor
                    for serial, kiosk in self.kiosks.items():
                        try:
                            response = await self.sio.emit('registration', {
                                'serial': serial,
                                'name': kiosk.name
                            }, callback=True)
                            
                            if response and response.get('status') == 'registered':
                                self.registered_kiosks[serial] = True
                                logger.info(f"Kiosk registrado: {kiosk.name} ({serial})")
                            else:
                                logger.warning(f"Kiosk no autorizado: {kiosk.name} ({serial})")
                                
                        except Exception as e:
                            logger.error(f"Error registrando kiosk {kiosk.name}: {str(e)}")
                            continue
                    
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Error conectando al servidor después de {max_retries} intentos: {str(e)}")
                        raise
                    logger.warning(f"Intento {retry_count} fallido, reintentando en 2 segundos...")
                    await asyncio.sleep(2)
            
            self.running = True
            logger.info(f"Spawner iniciado con {len(self.kiosks)} kiosks")
            
        except Exception as e:
            logger.error(f"Error iniciando kiosks: {str(e)}")
            raise
            
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
        
    async def on_disconnect(self):
        """Manejador de evento de desconexión"""
        logger.info("❌ Desconectado del servidor central")
        self.registered_kiosks.clear()  # Limpiar registro al desconectar
        
    async def handle_command(self, data):
        """Procesa comandos recibidos del servidor central"""
        try:
            kiosk_id = data.get('kiosk_id')
            if not kiosk_id:
                logger.error(f"Comando sin kiosk_id: {data}")
                return
                
            logger.info(f"📥 Comando recibido para kiosk {kiosk_id}")
            
            if kiosk_id in self.kiosks:
                kiosk = self.kiosks[kiosk_id]
                response = kiosk.process_command(data)
                await self.sio.emit('command_response', response)
                logger.info(f"📤 Respuesta enviada para comando {data.get('command_id')}")
            else:
                logger.error(f"Kiosk no encontrado: {kiosk_id}")
                
        except Exception as e:
            logger.error(f"Error procesando comando: {str(e)}")
            
    async def start_kiosks(self):
        """Inicia el ciclo de envío de datos de los kiosks"""
        try:
            while self.running:
                for serial, kiosk in self.kiosks.items():
                    # Solo enviar datos si el kiosk está registrado
                    if serial in self.registered_kiosks and self.registered_kiosks[serial]:
                        try:
                            telemetry = kiosk.get_telemetry_data()
                            # Formatear datos según el esquema esperado
                            data = {
                                'serial': serial,
                                'status': {
                                    'current': telemetry.get('status', 'unknown')
                                },
                                'sensors': {
                                    'temperature': {
                                        'value': telemetry.get('temperature', 0)
                                    },
                                    'humidity': {
                                        'value': telemetry.get('humidity', 0)
                                    },
                                    'door_status': telemetry.get('door_status', 'unknown'),
                                    'printer_status': telemetry.get('printer_status', 'unknown'),
                                    'network_quality': {
                                        'value': telemetry.get('network_quality', 0)
                                    },
                                    'voltage': telemetry.get('voltage', 220),
                                    'ventilation': telemetry.get('ventilation', 'normal')
                                }
                            }
                            await self.sio.emit('kiosk_update', data)
                            logger.debug(f"Telemetría enviada para kiosk {serial}")
                        except Exception as e:
                            logger.error(f"Error enviando datos de kiosk {serial}: {str(e)}")
                await asyncio.sleep(5)  # Actualizar cada 5 segundos
        except Exception as e:
            logger.error(f"Error en ciclo de datos: {str(e)}")
            self.running = False
            
    def start(self):
        """Punto de entrada principal"""
        try:
            async def main():
                await self.initialize()
                await self.start_kiosks()
                await self.cleanup()
                
            asyncio.run(main())
            
        except KeyboardInterrupt:
            logger.info("Spawner detenido por usuario")
        except Exception as e:
            logger.error(f"Error fatal: {str(e)}")
            raise
            
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inicia múltiples kiosks simulados')
    parser.add_argument('--config-dir', default='kiosk_configs',
                      help='Directorio con configuraciones de kiosks')
    parser.add_argument('--server-url', default='http://localhost:5000',
                      help='URL del servidor central')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    spawner = KioskSpawner(config_dir=args.config_dir, server_url=args.server_url)
    spawner.start() 