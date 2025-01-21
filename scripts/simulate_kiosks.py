"""
Script para simular datos de kiosks en tiempo real en Paraguay.
Este script genera datos simulados de kiosks y los envía al servidor
para probar la funcionalidad de WebSockets y monitoreo en tiempo real.
"""

import time
import random
from datetime import datetime
import logging
from typing import Dict, Any
import json
import threading
from dataclasses import dataclass
import socketio

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cliente SocketIO global
sio = socketio.Client(logger=True, engineio_logger=True)

@sio.event
def connect():
    """Manejador del evento de conexión"""
    logger.info("Conectado al servidor WebSocket")

@sio.event
def connect_error(data):
    """Manejador de errores de conexión"""
    logger.error(f"Error de conexión: {data}")

@sio.event
def disconnect():
    """Manejador del evento de desconexión"""
    logger.info("Desconectado del servidor WebSocket")

@dataclass
class Location:
    latitude: float
    longitude: float
    altitude: float
    accuracy: float

    def drift(self):
        """Simula pequeños cambios en la ubicación GPS"""
        self.latitude += random.uniform(-0.0001, 0.0001)
        self.longitude += random.uniform(-0.0001, 0.0001)
        self.accuracy = min(100, max(1, self.accuracy + random.uniform(-1, 1)))

class KioskSimulator:
    def __init__(self, kiosk_id: int, name: str, base_url: str = "http://localhost:5000"):
        self.kiosk_id = kiosk_id
        self.name = name
        self.base_url = base_url
        
        # Estado del kiosk
        self.status = 'active'
        self.anomaly_mode = False
        self.anomaly_counter = 0
        self.connection_quality = 1.0  # 1.0 = perfecta, 0.0 = sin conexión
        
        # Hardware simulado (adaptado a condiciones de Paraguay)
        self.cpu_model = random.choice([
            "Intel Core i3-10100", "Intel Core i5-10400", 
            "AMD Ryzen 3 3200G", "AMD Ryzen 5 3400G"
        ])
        self.ram_total = random.choice([4096, 8192, 16384])  # MB
        self.storage_total = random.choice([128, 256, 512])  # GB
        
        # Ubicación inicial (coordenadas de diferentes lugares en Paraguay)
        self.location = Location(
            latitude=random.uniform(-25.3867, -25.2667),  # Asunción y alrededores
            longitude=random.uniform(-57.3333, -57.2833),
            altitude=random.uniform(43, 250),  # Altitud típica de Asunción
            accuracy=random.uniform(5, 15)
        )
        
        # Patrones de uso adaptados a horarios locales de Paraguay
        self.usage_pattern = self._generate_usage_pattern()
        
    def _generate_usage_pattern(self) -> Dict[int, float]:
        """Genera un patrón de uso basado en la hora del día en Paraguay"""
        pattern = {}
        for hour in range(24):
            if 0 <= hour < 5:  # Madrugada
                pattern[hour] = 0.1
            elif 5 <= hour < 7:  # Inicio temprano
                pattern[hour] = 0.4
            elif 7 <= hour < 11:  # Mañana (alto uso)
                pattern[hour] = 0.9
            elif 11 <= hour < 15:  # Siesta
                pattern[hour] = 0.5
            elif 15 <= hour < 19:  # Tarde (alto uso)
                pattern[hour] = 0.95
            elif 19 <= hour < 22:  # Noche
                pattern[hour] = 0.7
            else:  # Noche tardía
                pattern[hour] = 0.3
        return pattern

    def generate_metrics(self) -> Dict[str, Any]:
        """Genera métricas realistas basadas en patrones de uso y estado"""
        current_hour = datetime.now().hour
        usage_multiplier = self.usage_pattern[current_hour] * self.connection_quality
        
        # Métricas base adaptadas al clima de Paraguay
        base_metrics = {
            "cpu_usage": self._generate_cpu_usage(usage_multiplier),
            "memory_usage": self._generate_memory_usage(usage_multiplier),
            "network_latency": self._generate_network_latency(),
            "storage_usage": self._generate_storage_usage(),
            "temperature": self._generate_temperature(),
            "timestamp": datetime.now().isoformat(),
            "location": {
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
                "altitude": self.location.altitude,
                "accuracy": self.location.accuracy
            },
            "hardware_info": {
                "cpu_model": self.cpu_model,
                "ram_total": self.ram_total,
                "storage_total": self.storage_total
            },
            "status": self.status,
            "connection_quality": self.connection_quality
        }
        
        if self.anomaly_mode:
            self._apply_anomaly_effects(base_metrics)
            
        return base_metrics

    def _generate_cpu_usage(self, multiplier: float) -> float:
        # Ajustado para clima cálido de Paraguay
        base = 35 if not self.anomaly_mode else 90
        variation = random.uniform(-5, 5)
        return min(100, max(0, (base + variation) * multiplier))

    def _generate_memory_usage(self, multiplier: float) -> float:
        base = 45 if not self.anomaly_mode else 95
        variation = random.uniform(-3, 3)
        return min(100, max(0, (base + variation) * multiplier))

    def _generate_network_latency(self) -> float:
        # Ajustado a condiciones de red en Paraguay
        base = random.uniform(20, 80) if not self.anomaly_mode else random.uniform(150, 300)
        return base / self.connection_quality if self.connection_quality > 0 else float('inf')

    def _generate_storage_usage(self) -> float:
        return min(100, max(0, random.uniform(50, 65)))

    def _generate_temperature(self) -> float:
        # Adaptado al clima de Paraguay (más cálido)
        hora_actual = datetime.now().hour
        temp_base = 40 if 11 <= hora_actual <= 15 else 35  # Más calor durante siesta
        if self.anomaly_mode:
            temp_base += 15
        return temp_base + random.uniform(-2, 2)

    def _apply_anomaly_effects(self, metrics: Dict[str, Any]):
        """Aplica efectos de anomalía a las métricas"""
        anomaly_type = random.choice(['cpu', 'memory', 'network', 'temperature'])
        if anomaly_type == 'cpu':
            metrics['cpu_usage'] *= 1.5
        elif anomaly_type == 'memory':
            metrics['memory_usage'] *= 1.4
        elif anomaly_type == 'network':
            metrics['network_latency'] *= 2.5  # Mayor impacto en red
        else:
            metrics['temperature'] *= 1.3

    def update_status(self):
        """Actualiza el estado del kiosk basado en condiciones"""
        if self.connection_quality < 0.2:
            self.status = 'offline'
        elif self.anomaly_mode:
            self.status = 'warning'
        else:
            self.status = 'active'

    def simulate_connection_changes(self):
        """Simula cambios en la calidad de conexión"""
        # Más variación en la calidad de conexión
        self.connection_quality = max(0, min(1, 
            self.connection_quality + random.uniform(-0.15, 0.15)))

    def run(self):
        """Ejecuta el simulador"""
        while True:
            try:
                # Simular cambios de conexión
                if random.random() < 0.15:  # 15% de probabilidad
                    self.simulate_connection_changes()
                
                # Simular anomalías (más frecuentes en clima cálido)
                if random.random() < 0.08:  # 8% de probabilidad
                    self.anomaly_mode = not self.anomaly_mode
                    logger.info(f"Kiosk {self.name}: {'Activando' if self.anomaly_mode else 'Desactivando'} modo anomalía")
                
                # Actualizar ubicación
                self.location.drift()
                
                # Actualizar estado
                self.update_status()
                
                # Generar y enviar métricas
                metrics = self.generate_metrics()
                self.send_metrics(metrics)
                
                # Intervalo aleatorio entre actualizaciones
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error en kiosk {self.name}: {str(e)}")
                time.sleep(5)  # Esperar antes de reintentar

    def send_metrics(self, metrics: Dict[str, Any]):
        """Envía métricas al servidor vía WebSocket"""
        try:
            # Emitir evento de métricas vía WebSocket
            sio.emit('metrics_update', {
                'kiosk_id': self.kiosk_id,
                'kiosk_name': self.name,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            logger.debug(f"Métricas enviadas para kiosk {self.name}")
        except Exception as e:
            logger.error(f"Error de conexión para {self.name}: {str(e)}")
            # Intentar reconectar si hay error
            try:
                if not sio.connected:
                    sio.connect(self.base_url)
            except Exception as reconnect_error:
                logger.error(f"Error al reconectar: {str(reconnect_error)}")

def main():
    """Función principal"""
    # Ubicaciones en Paraguay
    locations = [
        "Shopping del Sol", "Paseo La Galería", "Shopping Mariscal López",
        "Shopping Villa Morra", "Multiplaza", "Shopping Pinedo",
        "Terminal de Ómnibus", "Plaza de los Héroes", "Mercado 4",
        "Shopping San Lorenzo", "Plaza Uruguaya", "Mercado Municipal de Luque",
        "Shopping Mcal. López", "Plaza Infante Rivarola", "Costanera de Asunción",
        "Shopping París", "Plaza de la Democracia", "Terminal de Ciudad del Este",
        "Shopping China", "Mercado Municipal de Fernando", "Plaza Juan E. O'Leary",
        "Shopping Zuni", "Plaza Italia", "Mercado Central de Abasto",
        "Shopping Five Stars", "Plaza de las Américas", "Terminal de San Lorenzo",
        "Shopping Continental", "Plaza Uruguaya", "Mercado 4 de Asunción"
    ]
    
    try:
        # Conectar al servidor WebSocket
        sio.connect('http://localhost:5000')
        
        simulators = [
            KioskSimulator(
                kiosk_id=i+1,
                name=f"Kiosk {locations[i]}"
            ) for i in range(30)
        ]
        
        # Iniciar simulación en threads separados
        threads = []
        for simulator in simulators:
            thread = threading.Thread(target=simulator.run)
            thread.daemon = True
            threads.append(thread)
            thread.start()
            logger.info(f"Iniciado simulador para {simulator.name}")
            
        try:
            while True:
                time.sleep(1)
                # Verificar conexión WebSocket
                if not sio.connected:
                    logger.warning("Conexión WebSocket perdida, intentando reconectar...")
                    try:
                        sio.connect('http://localhost:5000')
                    except Exception as e:
                        logger.error(f"Error al reconectar: {str(e)}")
                        
        except KeyboardInterrupt:
            logger.info("Deteniendo simulación...")
            sio.disconnect()
            
    except Exception as e:
        logger.error(f"Error al iniciar simulación: {str(e)}")
        if sio.connected:
            sio.disconnect()

if __name__ == "__main__":
    main() 