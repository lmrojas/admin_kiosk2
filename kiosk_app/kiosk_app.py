"""
Módulo para envío de datos del kiosk
"""

import os
import logging
import platform
import psutil
import netifaces
import requests
from datetime import datetime
import time
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from kiosk_behavior_simulator import (
    KioskBehaviorSimulator, 
    NetworkSimConfig,
    HardwareSimConfig,
    SensorSimConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KioskConfig:
    """Configuración básica de un kiosk"""
    serial: str
    name: str
    location: dict
    timezone: str = "UTC"  # Zona horaria por defecto

class Kiosk:
    """Simulador de kiosk que envía datos al servidor central"""
    
    def __init__(self, config: KioskConfig):
        """Inicializa el simulador con la configuración proporcionada"""
        self.config = config
        self.timezone = ZoneInfo(config.timezone)
        self.is_registered = False
        self.socket_id = None
        
        # Configuración inicial para el simulador de comportamiento
        initial_config = {
            'network': NetworkSimConfig(
                base_signal_quality=95.0,
                base_latency=20.0,
                base_speed=100000000,
                packet_loss_probability=0.001
            ),
            'hardware': HardwareSimConfig(
                base_cpu_usage=30.0,
                base_memory_usage=40.0,
                base_cpu_temp=45.0
            ),
            'sensors': SensorSimConfig(
                base_temperature=25.0,
                base_humidity=60.0,
                base_voltage=220.0
            ),
            'location': config.location  # Agregando la ubicación desde la configuración del kiosk
        }
        
        self.behavior_simulator = KioskBehaviorSimulator(initial_config)
        self._initialize_hardware_info()
        
    def _initialize_hardware_info(self):
        """Inicializa la información de hardware del kiosk"""
        self.hardware_info = {
            'os': {
                'name': platform.system(),
                'version': platform.version(),
                'platform': platform.platform()
            },
            'memory': {
                'total': psutil.virtual_memory().total / (1024 ** 3)  # GB
            },
            'network': {
                'interfaces': [i for i in netifaces.interfaces() if i != 'lo'],
                'mac_address': self._get_mac_address()
            }
        }
    
    def _get_mac_address(self):
        """Obtiene la dirección MAC de la interfaz principal"""
        try:
            interfaces = netifaces.interfaces()
            for interface in interfaces:
                if interface != 'lo':
                    addrs = netifaces.ifaddresses(interface)
                    if netifaces.AF_LINK in addrs:
                        return addrs[netifaces.AF_LINK][0]['addr']
        except Exception as e:
            logger.error(f"Error obteniendo MAC address: {str(e)}")
            return None

    def get_registration_data(self) -> dict:
        """Obtiene los datos necesarios para registrar el kiosk"""
        return {
            'serial': self.config.serial,
            'name': self.config.name,
            'location': self.config.location,
            'timezone': self.config.timezone,
            'hardware_info': self.hardware_info,
            'registration_time': datetime.now(self.timezone).isoformat()
        }
        
    def get_telemetry_data(self) -> dict:
        """Obtiene los datos de telemetría simulados"""
        return self.behavior_simulator.get_simulated_data()
        
    def process_command(self, command: str, params: dict) -> dict:
        """Procesa un comando recibido del servidor"""
        logger.info(f"Procesando comando {command} con parámetros {params}")
        # Aquí se implementaría la lógica para procesar diferentes comandos
        return {
            'status': 'success',
            'message': f'Comando {command} procesado correctamente'
        }

    def get_telemetry(self):
        """Obtiene datos de telemetría simulados"""
        simulated_data = self.behavior_simulator.get_simulated_data()
        time_info = self._get_local_time()
        
        return {
            # Identificación
            "serial": self.config.serial,
            "name": self.config.name,
            "network": self._get_network_info(),
            
            # Estado del Sistema
            "system_status": {
                "os_name": platform.system(),
                "os_version": platform.version(),
                "uptime": time.time(),
                "local_time": time_info["local_time"],
                "timezone": time_info["timezone"],
                "utc_offset": time_info["utc_offset"]
            },
            
            # Recursos Hardware
            "hardware": simulated_data["hardware"],
            
            # Sensores
            "sensors": simulated_data["sensors"],
            
            # Geolocalización
            "geolocation": self._get_geolocation(),
            
            # Timestamp UTC
            "timestamp": datetime.now(ZoneInfo("UTC")).isoformat()
        }
    
    def _get_local_time(self) -> dict:
        """Obtiene información de hora local"""
        now = datetime.now(self.timezone)
        return {
            "local_time": now.isoformat(),
            "timezone": str(self.timezone),
            "utc_offset": now.utcoffset().total_seconds() / 3600
        }
        
    def _get_network_info(self) -> dict:
        """Obtiene información de red"""
        try:
            interfaces = netifaces.interfaces()
            active_interface = None
            
            for iface in interfaces:
                if iface != 'lo':
                    addrs = netifaces.ifaddresses(iface)
                    if netifaces.AF_INET in addrs:
                        active_interface = iface
                        break
            
            if active_interface:
                addrs = netifaces.ifaddresses(active_interface)
                ipv4 = addrs[netifaces.AF_INET][0]
                mac = addrs[netifaces.AF_LINK][0]['addr'] if netifaces.AF_LINK in addrs else None
                
                # Obtener IP pública
                try:
                    public_ip = requests.get('https://api.ipify.org').text
                except:
                    public_ip = None
                
                # Obtener métricas simuladas
                network_metrics = self.behavior_simulator.get_simulated_data()["network"]
                
                return {
                    "local_ip": ipv4.get('addr'),
                    "public_ip": public_ip,
                    "mac_address": mac,
                    "interface": active_interface,
                    "signal_quality": network_metrics["signal_quality"],
                    "connection_speed": network_metrics["connection_speed"],
                    "latency": network_metrics["latency"],
                    "connection_status": network_metrics["connection_status"],
                    "packets": {
                        "sent": network_metrics["packets_sent"],
                        "received": network_metrics["packets_received"],
                        "lost": network_metrics["packets_lost"]
                    }
                }
            return {"error": "No active network interface found"}
        except Exception as e:
            logger.error(f"Error getting network info: {str(e)}")
            return {"error": str(e)}

    def _get_geolocation(self) -> dict:
        """Obtiene información de geolocalización"""
        geo_data = self.behavior_simulator.get_simulated_data()["geolocation"]
        return {
            "latitude": geo_data["latitude"],
            "longitude": geo_data["longitude"],
            "timestamp": datetime.now(self.timezone).isoformat()
        } 