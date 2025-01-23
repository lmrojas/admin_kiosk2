"""
Simulador de comportamiento para el kiosk
Genera variaciones realistas en los datos enviados
"""

import random
import time
import math
from datetime import datetime, timedelta
import ipaddress
from dataclasses import dataclass

@dataclass
class NetworkSimConfig:
    """Configuración para simulación de red"""
    base_signal_quality: float = 95.0
    base_latency: float = 20.0
    base_speed: int = 100000000  # 100 Mbps
    packet_loss_probability: float = 0.001

@dataclass
class HardwareSimConfig:
    """Configuración para simulación de hardware"""
    base_cpu_usage: float = 30.0
    base_memory_usage: float = 40.0
    base_cpu_temp: float = 45.0

@dataclass
class SensorSimConfig:
    """Configuración para simulación de sensores"""
    base_temperature: float = 25.0
    base_humidity: float = 60.0
    base_voltage: float = 220.0

class NetworkSimulator:
    """Simula comportamiento de red"""
    
    def __init__(self, config: NetworkSimConfig):
        self.config = config
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_lost = 0
        self._last_update = time.time()
        
    def get_network_metrics(self) -> dict:
        """Genera métricas de red realistas"""
        current_time = time.time()
        time_diff = current_time - self._last_update
        
        # Simula degradación gradual de señal
        noise = math.sin(current_time * 0.1) * 5
        signal_quality = max(0, min(100, self.config.base_signal_quality + noise))
        
        # Simula variación en latencia
        latency_variation = random.gauss(0, 5)
        latency = max(1, self.config.base_latency + latency_variation)
        
        # Simula fluctuaciones en velocidad
        speed_noise = random.uniform(0.8, 1.2)
        connection_speed = int(self.config.base_speed * speed_noise)
        
        # Actualiza contadores de paquetes
        new_packets = int(time_diff * 10)  # 10 paquetes por segundo
        self.packets_sent += new_packets
        lost_packets = int(new_packets * self.config.packet_loss_probability)
        self.packets_lost += lost_packets
        self.packets_received += (new_packets - lost_packets)
        
        self._last_update = current_time
        
        return {
            "signal_quality": signal_quality,
            "connection_speed": connection_speed,
            "latency": latency,
            "connection_status": "connected",
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "packets_lost": self.packets_lost
        }

class HardwareSimulator:
    """Simula comportamiento de hardware"""
    
    def __init__(self, config: HardwareSimConfig):
        self.config = config
        self._last_update = time.time()
        self._cpu_trend = 0
        self._memory_trend = 0
        
    def get_hardware_metrics(self) -> dict:
        """Genera métricas de hardware realistas"""
        current_time = time.time()
        
        # Simula tendencias en uso de CPU
        self._cpu_trend += random.uniform(-0.1, 0.1)
        self._cpu_trend = max(-1, min(1, self._cpu_trend))
        cpu_usage = max(0, min(100, self.config.base_cpu_usage + self._cpu_trend * 20))
        
        # Simula temperatura de CPU basada en uso
        cpu_temp = self.config.base_cpu_temp + (cpu_usage - self.config.base_cpu_usage) * 0.1
        
        # Simula uso de memoria con tendencia
        self._memory_trend += random.uniform(-0.05, 0.05)
        self._memory_trend = max(-1, min(1, self._memory_trend))
        memory_usage = max(0, min(100, self.config.base_memory_usage + self._memory_trend * 15))
        
        return {
            "cpu": {
                "usage": cpu_usage,
                "temperature": cpu_temp
            },
            "memory": {
                "percent": memory_usage
            }
        }

class SensorSimulator:
    """Simula comportamiento de sensores"""
    
    def __init__(self, config: SensorSimConfig):
        self.config = config
        self._last_update = time.time()
        self._temp_trend = 0
        self._humidity_trend = 0
        
    def get_sensor_data(self) -> dict:
        """Genera datos de sensores realistas"""
        current_time = time.time()
        
        # Simula variación en temperatura
        self._temp_trend += random.uniform(-0.05, 0.05)
        self._temp_trend = max(-1, min(1, self._temp_trend))
        temperature = self.config.base_temperature + self._temp_trend * 5
        
        # Simula variación en humedad
        self._humidity_trend += random.uniform(-0.02, 0.02)
        self._humidity_trend = max(-1, min(1, self._humidity_trend))
        humidity = max(30, min(90, self.config.base_humidity + self._humidity_trend * 10))
        
        # Simula pequeñas fluctuaciones en voltaje
        voltage = self.config.base_voltage + random.uniform(-2, 2)
        
        return {
            "temperature": temperature,
            "humidity": humidity,
            "voltage": voltage,
            "door": "closed"  # Podría cambiar basado en eventos
        }

class GeoSimulator:
    """Simula variaciones en geolocalización"""
    
    def __init__(self, base_lat: float, base_lon: float):
        self.base_lat = base_lat
        self.base_lon = base_lon
        self._last_update = time.time()
        
    def get_location(self) -> dict:
        """Genera variaciones realistas en coordenadas WiFi"""
        # Simula imprecisión de WiFi (±50 metros)
        lat_variation = random.uniform(-0.0004, 0.0004)
        lon_variation = random.uniform(-0.0004, 0.0004)
        
        return {
            "latitude": self.base_lat + lat_variation,
            "longitude": self.base_lon + lon_variation,
            "timestamp": datetime.now().isoformat()
        }

class KioskBehaviorSimulator:
    """Coordina la simulación de comportamiento del kiosk"""
    
    def __init__(self, initial_config: dict):
        self.network_sim = NetworkSimulator(NetworkSimConfig())
        self.hardware_sim = HardwareSimulator(HardwareSimConfig())
        self.sensor_sim = SensorSimulator(SensorSimConfig())
        self.geo_sim = GeoSimulator(
            initial_config["location"]["latitude"],
            initial_config["location"]["longitude"]
        )
    
    def get_simulated_data(self) -> dict:
        """Obtiene datos simulados para todos los aspectos del kiosk"""
        return {
            "network": self.network_sim.get_network_metrics(),
            "hardware": self.hardware_sim.get_hardware_metrics(),
            "sensors": self.sensor_sim.get_sensor_data(),
            "geolocation": self.geo_sim.get_location()
        } 