# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import db
from datetime import datetime
import uuid
from sqlalchemy.orm import relationship
from flask_login import current_user
from app.models.kiosk_location_history import KioskLocationHistory

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
    store_name = db.Column(db.String(200), nullable=True)
    location = db.Column(db.String(200), nullable=True)
    
    # Información de Geolocalización
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    altitude = db.Column(db.Float, nullable=True)
    location_updated_at = db.Column(db.DateTime, nullable=True)
    location_accuracy = db.Column(db.Float, nullable=True)  # precisión en metros
    
    # Ubicación reportada por el kiosk
    reported_latitude = db.Column(db.Float, nullable=True)
    reported_longitude = db.Column(db.Float, nullable=True)
    reported_altitude = db.Column(db.Float, nullable=True)
    reported_location_updated_at = db.Column(db.DateTime, nullable=True)
    reported_location_accuracy = db.Column(db.Float, nullable=True)
    
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
    
    # Capacidades y Credenciales
    capabilities = db.Column(db.JSON, nullable=True)
    credentials_hash = db.Column(db.String(128), nullable=True)
    
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
    
    # Nuevo campo
    socket_id = db.Column(db.String(50))  # ID de sesión WebSocket
    
    def update_location(self, latitude, longitude, altitude=None, accuracy=None):
        """Actualiza la ubicación asignada del kiosk."""
        # Guardar ubicación anterior
        prev_lat = self.latitude
        prev_lon = self.longitude
        
        # Actualizar ubicación actual
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.location_accuracy = accuracy
        self.location_updated_at = datetime.utcnow()
        
        # Registrar en historial
        history = KioskLocationHistory(
            kiosk_id=self.id,
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            timestamp=self.location_updated_at,
            location_type='assigned',
            previous_latitude=prev_lat,
            previous_longitude=prev_lon,
            change_reason='Actualización manual de ubicación',
            created_by=current_user.id if current_user and not current_user.is_anonymous else None
        )
        db.session.add(history)
    
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
    
    def validate_credentials(self, credentials: dict) -> bool:
        """
        Valida las credenciales proporcionadas contra las almacenadas.
        Args:
            credentials (dict): Credenciales a validar
        Returns:
            bool: True si las credenciales son válidas
        """
        if not self.credentials_hash:
            return False
        
        # Aquí implementar la lógica de validación usando hash seguro
        # Por ahora retornamos True para testing
        return True
    
    def __repr__(self):
        """
        Representación en cadena del modelo para depuración.
        No incluye lógica de negocio compleja.
        """
        return f'<Kiosk {self.name} (UUID: {self.uuid}, Health: {self.health_score:.2f})>'

    def to_dict(self):
        """
        Convierte el kiosk a un diccionario para la API.
        No incluye lógica de negocio compleja.
        """
        return {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'location': self.location,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'location_accuracy': self.location_accuracy,
            'location_updated_at': self.location_updated_at.isoformat() if self.location_updated_at else None,
            'status': self.status,
            'last_online': self.last_online.isoformat() if self.last_online else None,
            'health_score': self.health_score,
            'anomaly_probability': self.anomaly_probability,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'socket_id': self.socket_id
        }

    def update_reported_location(self, latitude, longitude, altitude=None, accuracy=None):
        """Actualiza la ubicación reportada por el kiosk."""
        # Guardar ubicación anterior
        prev_lat = self.reported_latitude
        prev_lon = self.reported_longitude
        
        # Actualizar ubicación actual
        self.reported_latitude = latitude
        self.reported_longitude = longitude
        self.reported_altitude = altitude
        self.reported_location_accuracy = accuracy
        self.reported_location_updated_at = datetime.utcnow()
        
        # Registrar en historial
        history = KioskLocationHistory(
            kiosk_id=self.id,
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            timestamp=self.reported_location_updated_at,
            location_type='reported',
            previous_latitude=prev_lat,
            previous_longitude=prev_lon,
            change_reason='Ubicación reportada por el kiosk',
            created_by=None  # Reportado automáticamente por el kiosk
        )
        db.session.add(history)

    def get_location_difference(self):
        """
        Calcula la diferencia entre la ubicación asignada y la reportada.
        Retorna la distancia en metros y la diferencia de tiempo.
        """
        if not all([self.latitude, self.longitude, self.reported_latitude, self.reported_longitude]):
            return None, None

        from app.services.geolocation_service import GeolocationService
        geo_service = GeolocationService()
        
        distance = geo_service.calculate_distance(
            self.latitude, self.longitude,
            self.reported_latitude, self.reported_longitude
        )
        
        time_diff = None
        if self.location_updated_at and self.reported_location_updated_at:
            time_diff = abs((self.reported_location_updated_at - self.location_updated_at).total_seconds())
        
        return distance, time_diff

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