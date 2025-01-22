"""
Script para simular comportamientos anómalos en los kiosks.
"""
import random
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AnomalySimulator:
    """Clase para simular comportamientos anómalos en kiosks"""
    
    def __init__(self):
        self.anomaly_types = {
            'cpu_spike': self._simulate_cpu_spike,
            'memory_leak': self._simulate_memory_leak,
            'network_latency': self._simulate_network_latency,
            'location_drift': self._simulate_location_drift,
            'connection_loss': self._simulate_connection_loss
        }
        self.active_anomalies: Dict[str, Dict[str, Any]] = {}
        
    def _simulate_cpu_spike(self, kiosk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula un pico de uso de CPU"""
        sensors = kiosk_data['sensors']
        sensors['cpu_usage']['value'] = random.uniform(85, 100)
        return kiosk_data
        
    def _simulate_memory_leak(self, kiosk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula una fuga de memoria gradual"""
        sensors = kiosk_data['sensors']
        current = sensors['memory_usage']['value']
        sensors['memory_usage']['value'] = min(100, current + random.uniform(2, 5))
        return kiosk_data
        
    def _simulate_network_latency(self, kiosk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula latencia de red alta"""
        sensors = kiosk_data['sensors']
        sensors['network_latency']['value'] = random.uniform(150, 500)
        return kiosk_data
        
    def _simulate_location_drift(self, kiosk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula deriva gradual en la ubicación reportada"""
        if 'coordinates' in kiosk_data:
            coords = kiosk_data['coordinates']
            if coords['reported']['latitude'] and coords['reported']['longitude']:
                # Deriva máxima de ~100 metros
                drift_lat = random.uniform(-0.0009, 0.0009)
                drift_lon = random.uniform(-0.0009, 0.0009)
                
                coords['reported']['latitude'] += drift_lat
                coords['reported']['longitude'] += drift_lon
                
        return kiosk_data
        
    def _simulate_connection_loss(self, kiosk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simula pérdida intermitente de conexión"""
        if random.random() < 0.3:  # 30% de probabilidad de desconexión
            kiosk_data['status'] = 'offline'
        return kiosk_data
        
    def start_anomaly(self, kiosk_id: str, anomaly_type: str, duration_minutes: int = 30):
        """Inicia una anomalía para un kiosk específico"""
        if anomaly_type not in self.anomaly_types:
            raise ValueError(f"Tipo de anomalía no válido: {anomaly_type}")
            
        self.active_anomalies[kiosk_id] = {
            'type': anomaly_type,
            'start_time': datetime.utcnow(),
            'end_time': datetime.utcnow() + timedelta(minutes=duration_minutes)
        }
        logger.info(f"Iniciada anomalía {anomaly_type} en kiosk {kiosk_id}")
        
    def stop_anomaly(self, kiosk_id: str):
        """Detiene una anomalía activa"""
        if kiosk_id in self.active_anomalies:
            anomaly = self.active_anomalies.pop(kiosk_id)
            logger.info(f"Detenida anomalía {anomaly['type']} en kiosk {kiosk_id}")
            
    def apply_anomalies(self, kiosk_id: str, kiosk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica anomalías activas a los datos del kiosk"""
        if kiosk_id not in self.active_anomalies:
            return kiosk_data
            
        anomaly = self.active_anomalies[kiosk_id]
        now = datetime.utcnow()
        
        # Verificar si la anomalía debe terminar
        if now > anomaly['end_time']:
            self.stop_anomaly(kiosk_id)
            return kiosk_data
            
        # Aplicar la anomalía
        simulator = self.anomaly_types[anomaly['type']]
        return simulator(kiosk_data)
        
    def random_anomaly(self, kiosk_id: str):
        """Inicia una anomalía aleatoria"""
        anomaly_type = random.choice(list(self.anomaly_types.keys()))
        duration = random.randint(5, 60)  # Entre 5 y 60 minutos
        self.start_anomaly(kiosk_id, anomaly_type, duration) 