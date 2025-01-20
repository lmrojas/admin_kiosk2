# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import db
from datetime import datetime
import uuid
from sqlalchemy.orm import relationship

class Kiosk(db.Model):
    """
    Modelo de Kiosk que representa un dispositivo en el sistema.
    Sigue el patrón MVT, sin incluir lógica de negocio compleja.
    """
    
    __tablename__ = 'kiosks'

    # Identificadores
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Información básica
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=True)
    
    # Información de Geolocalización
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    altitude = db.Column(db.Float, nullable=True)
    location_updated_at = db.Column(db.DateTime, nullable=True)
    location_accuracy = db.Column(db.Float, nullable=True)  # precisión en metros
    
    # Estado del Kiosk
    status = db.Column(db.String(20), default='inactive')
    last_online = db.Column(db.DateTime, nullable=True)
    
    # Información de Hardware
    cpu_model = db.Column(db.String(100), nullable=True)
    ram_total = db.Column(db.Float, nullable=True)  # en GB
    storage_total = db.Column(db.Float, nullable=True)  # en GB
    
    # Información de Red
    ip_address = db.Column(db.String(45), nullable=True)
    mac_address = db.Column(db.String(17), nullable=True)
    
    # Metadatos
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relación con Usuario (opcional)
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    owner = relationship('User', foreign_keys=[owner_id], back_populates='owned_kiosks')
    
    # Métricas de IA y Salud
    health_score = db.Column(db.Float, default=100.0)
    anomaly_probability = db.Column(db.Float, default=0.0)
    
    # Relaciones
    sensor_data = relationship('SensorData', back_populates='kiosk', lazy='dynamic')
    
    def update_location(self, latitude, longitude, altitude=None, accuracy=None):
        """
        Actualiza la información de geolocalización del kiosk.
        La lógica compleja debe ir en el servicio correspondiente.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.location_accuracy = accuracy
        self.location_updated_at = datetime.utcnow()
    
    def calculate_health_score(self):
        """
        Calcula el score de salud basado en métricas.
        NOTA: La lógica compleja debe ir en un servicio.
        """
        if not self.sensor_data:
            return 100.0
        
        # Ejemplo simple de cálculo
        latest_data = self.sensor_data.order_by(SensorData.timestamp.desc()).first()
        if latest_data:
            # Lógica básica de cálculo
            score = 100.0
            if latest_data.cpu_usage > 90:
                score -= 20
            if latest_data.memory_usage > 90:
                score -= 20
            return max(0.0, score)
        
        return 100.0
    
    def update_health_metrics(self, anomaly_prob):
        """
        Actualiza métricas de salud y anomalías.
        """
        self.health_score = self.calculate_health_score()
        self.anomaly_probability = anomaly_prob
    
    def __repr__(self):
        """
        Representación en cadena del modelo para depuración.
        No incluye lógica de negocio compleja.
        """
        return f'<Kiosk {self.name} (UUID: {self.uuid}, Health: {self.health_score:.2f})>'

# Modelo de SensorData para complementar Kiosk
class SensorData(db.Model):
    """
    Modelo para almacenar datos de sensores de un Kiosk.
    Sigue el patrón MVT, sin lógica de negocio compleja.
    """
    
    __tablename__ = 'sensor_data'
    
    id = db.Column(db.Integer, primary_key=True)
    kiosk_id = db.Column(db.Integer, db.ForeignKey('kiosks.id'), nullable=False)
    
    # Métricas de sistema
    cpu_usage = db.Column(db.Float, nullable=False)
    memory_usage = db.Column(db.Float, nullable=False)
    network_latency = db.Column(db.Float, nullable=True)
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relación con Kiosk
    kiosk = relationship('Kiosk', back_populates='sensor_data')
    
    def __repr__(self):
        return f'<SensorData for Kiosk {self.kiosk_id} at {self.timestamp}>' 