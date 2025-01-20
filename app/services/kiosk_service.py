# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import db
from app.models.kiosk import Kiosk, SensorData
from app.services.kiosk_ai_service import KioskAIService
from datetime import datetime, timedelta
import uuid

class KioskService:
    """
    Servicio para manejar la lógica de negocio relacionada con Kiosks.
    Sigue el patrón de Services, separando la lógica de los modelos.
    """

    @staticmethod
    def create_kiosk(name, location=None, owner_id=None):
        """
        Crea un nuevo kiosk con validaciones básicas.
        """
        if not name:
            raise ValueError("El nombre del kiosk es obligatorio")

        kiosk = Kiosk(
            name=name, 
            location=location, 
            owner_id=owner_id,
            status='inactive',
            uuid=str(uuid.uuid4())
        )
        
        db.session.add(kiosk)
        db.session.commit()
        return kiosk

    @staticmethod
    def update_kiosk_status(kiosk_id, status, hardware_info=None):
        """
        Actualiza el estado de un kiosk.
        """
        kiosk = Kiosk.query.get(kiosk_id)
        if not kiosk:
            raise ValueError(f"Kiosk con ID {kiosk_id} no encontrado")

        kiosk.status = status
        kiosk.last_online = datetime.utcnow()

        if hardware_info:
            kiosk.cpu_model = hardware_info.get('cpu_model', kiosk.cpu_model)
            kiosk.ram_total = hardware_info.get('ram_total', kiosk.ram_total)
            kiosk.storage_total = hardware_info.get('storage_total', kiosk.storage_total)
            kiosk.ip_address = hardware_info.get('ip_address', kiosk.ip_address)
            kiosk.mac_address = hardware_info.get('mac_address', kiosk.mac_address)

        db.session.commit()
        return kiosk

    @staticmethod
    def register_sensor_data(kiosk_id, cpu_usage, memory_usage, network_latency=None):
        """
        Registra datos de sensores para un kiosk y actualiza su estado de salud.
        """
        kiosk = Kiosk.query.get(kiosk_id)
        if not kiosk:
            raise ValueError(f"Kiosk con ID {kiosk_id} no encontrado")

        # Crear registro de sensor data
        sensor_data = SensorData(
            kiosk_id=kiosk_id,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_latency=network_latency
        )
        db.session.add(sensor_data)

        # Usar servicio de IA para predecir anomalías
        ai_service = KioskAIService()
        anomaly_prob = ai_service.predict_anomaly({
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'network_latency': network_latency or 0
        })

        # Actualizar métricas de salud del kiosk
        kiosk.update_health_metrics(anomaly_prob)

        db.session.commit()
        return sensor_data

    @staticmethod
    def get_kiosks_by_health_status(min_health=50.0, max_health=100.0):
        """
        Obtiene kiosks filtrados por su score de salud.
        """
        return Kiosk.query.filter(
            Kiosk.health_score.between(min_health, max_health)
        ).all()

    @staticmethod
    def get_kiosks_with_recent_anomalies(hours=24):
        """
        Obtiene kiosks con anomalías recientes.
        """
        threshold_time = datetime.utcnow() - timedelta(hours=hours)
        return Kiosk.query.join(SensorData).filter(
            SensorData.timestamp >= threshold_time,
            Kiosk.anomaly_probability > 0.5
        ).distinct().all()

    @staticmethod
    def get_all_kiosks():
        """
        Obtiene todos los kiosks registrados en el sistema.
        Returns:
            List[Kiosk]: Lista de todos los kiosks
        """
        return Kiosk.query.all() 