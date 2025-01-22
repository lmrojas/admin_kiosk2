"""
Módulo para la simulación de kiosks individuales.
Maneja la lógica de generación de datos y comportamiento de cada kiosk.
"""
import os
import json
import random
import logging
from datetime import datetime
import psutil
import socket
import uuid
from typing import Dict, Any, Optional
from .anomaly_simulator import AnomalySimulator

logger = logging.getLogger(__name__)

class KioskApp:
    """Clase que simula el comportamiento de un kiosk individual"""
    
    # Estados válidos según el modelo de BD
    VALID_STATES = ['active', 'offline', 'blocked', 'maintenance', 'error']
    
    def __init__(self, config: Dict[str, Any]):
        self.serial = config['serial']
        self.name = config['name']
        self.location = config['location']
        self.status = config.get('status', 'offline')
        self.coordinates = config.get('coordinates', {
            'assigned': {'latitude': None, 'longitude': None},
            'reported': {'latitude': None, 'longitude': None}
        })
        self.last_update = datetime.utcnow()
        self.hardware_info = config.get('hardware', self._get_hardware_info())
        self.sensors = self._initialize_sensors()
        self.anomaly_simulator = AnomalySimulator()
        
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Obtiene información real del hardware"""
        return {
            'cpu_model': 'Simulated CPU',
            'ram_total': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
            'storage_total': psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # GB
            'ip_address': socket.gethostbyname(socket.gethostname()),
            'mac_address': uuid.getnode().to_bytes(6, 'big').hex(':')
        }
        
    def _initialize_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa los sensores con rangos realistas"""
        return {
            'temperature': {
                'value': random.uniform(18, 25),
                'unit': 'C',
                'min': 15,
                'max': 35,
                'variation': 0.5
            },
            'humidity': {
                'value': random.uniform(40, 60),
                'unit': '%',
                'min': 30,
                'max': 80,
                'variation': 2
            },
            'cpu_usage': {
                'value': random.uniform(10, 30),
                'unit': '%',
                'min': 0,
                'max': 100,
                'variation': 5
            },
            'memory_usage': {
                'value': random.uniform(20, 40),
                'unit': '%',
                'min': 0,
                'max': 100,
                'variation': 3
            },
            'network_latency': {
                'value': random.uniform(5, 50),
                'unit': 'ms',
                'min': 1,
                'max': 200,
                'variation': 10
            }
        }
        
    def update_sensors(self):
        """Actualiza todos los sensores"""
        for name, sensor in self.sensors.items():
            variation = random.uniform(-sensor['variation'], sensor['variation'])
            sensor['value'] = max(sensor['min'], min(sensor['max'], sensor['value'] + variation))
        self.last_update = datetime.utcnow()
        
    def get_telemetry_data(self) -> Dict[str, Any]:
        """Genera datos de telemetría según el formato esperado por el sistema central"""
        self.update_sensors()
        
        data = {
            'kiosk_id': self.serial,
            'name': self.name,
            'timestamp': self.last_update.isoformat(),
            'status': self.status,
            'location': self.location,
            'coordinates': self.coordinates,
            'hardware_info': self.hardware_info,
            'sensors': {name: {'value': sensor['value'], 'unit': sensor['unit']} 
                       for name, sensor in self.sensors.items()}
        }
        
        # Aplicar anomalías si existen
        return self.anomaly_simulator.apply_anomalies(self.serial, data)
        
    def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa comandos recibidos del sistema central"""
        command_type = command.get('type')
        command_id = command.get('command_id')
        
        response = {
            'command_id': command_id,
            'kiosk_id': self.serial,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'error',
            'message': 'Comando no reconocido'
        }
        
        if command_type == 'status_update':
            new_status = command.get('status')
            if new_status in self.VALID_STATES:
                self.status = new_status
                response.update({
                    'status': 'success', 
                    'message': f'Estado actualizado a: {self.status}'
                })
                
        elif command_type == 'get_telemetry':
            response.update({
                'status': 'success',
                'message': 'Telemetría obtenida',
                'data': self.get_telemetry_data()
            })
            
        elif command_type == 'start_anomaly':
            anomaly_type = command.get('anomaly_type')
            duration = command.get('duration_minutes', 30)
            try:
                self.anomaly_simulator.start_anomaly(self.serial, anomaly_type, duration)
                response.update({
                    'status': 'success',
                    'message': f'Anomalía {anomaly_type} iniciada'
                })
            except ValueError as e:
                response.update({
                    'message': str(e)
                })
                
        elif command_type == 'stop_anomaly':
            self.anomaly_simulator.stop_anomaly(self.serial)
            response.update({
                'status': 'success',
                'message': 'Anomalía detenida'
            })
            
        elif command_type == 'random_anomaly':
            self.anomaly_simulator.random_anomaly(self.serial)
            response.update({
                'status': 'success',
                'message': 'Anomalía aleatoria iniciada'
            })
            
        return response

    @classmethod
    def initialize_instances(cls, config_dir: str) -> Dict[str, 'KioskApp']:
        """Inicializa múltiples instancias de kiosks desde archivos de configuración"""
        instances = {}
        
        try:
            # Listar todos los archivos .json de configuración
            config_files = [f for f in os.listdir(config_dir) 
                          if f.endswith('.json') and f != 'kiosks_summary.json']
            
            logger.info(f"Encontrados {len(config_files)} archivos de configuración")
            
            for config_file in config_files:
                file_path = os.path.join(config_dir, config_file)
                try:
                    with open(file_path, 'r') as f:
                        kiosk_config = json.load(f)
                        if isinstance(kiosk_config, dict) and 'serial' in kiosk_config:
                            kiosk = cls(kiosk_config)
                            instances[kiosk.serial] = kiosk  # Usamos serial como clave
                            logger.info(f"Kiosk inicializado: {kiosk.name} ({kiosk.serial})")
                except Exception as e:
                    logger.error(f"Error leyendo configuración {config_file}: {str(e)}")
                    continue
                    
            return instances
            
        except Exception as e:
            logger.error(f"Error inicializando instancias: {str(e)}")
            return {} 