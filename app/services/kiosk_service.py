# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app import db
from app.models.kiosk import Kiosk, SensorData
from app.services.kiosk_ai_service import KioskAIService
from datetime import datetime, timedelta
import uuid
import math
import logging

logger = logging.getLogger(__name__)

class KioskService:
    """
    Servicio para manejar la lógica de negocio relacionada con Kiosks.
    Sigue el patrón de Services, separando la lógica de los modelos.
    """

    @staticmethod
    def verify_tables():
        """
        Verifica que las tablas necesarias para kiosks existan en la base de datos.
        No crea kiosks - solo verifica la estructura.
        
        Returns:
            bool: True si la estructura es correcta
            
        Raises:
            Exception: Si hay algún problema con la estructura
        """
        try:
            inspector = db.inspect(db.engine)
            
            # Verificar tabla de kiosks
            if 'kiosks' not in inspector.get_table_names():
                raise Exception("Tabla de kiosks no existe")
                
            # Verificar tabla de datos de sensores
            if 'sensor_data' not in inspector.get_table_names():
                raise Exception("Tabla de datos de sensores no existe")
                
            # Verificar columnas requeridas en Kiosk
            kiosk_columns = [c['name'] for c in inspector.get_columns('kiosks')]
            required_columns = ['id', 'name', 'uuid', 'status', 'owner_id', 'capabilities', 'credentials_hash']
            missing = [col for col in required_columns if col not in kiosk_columns]
            if missing:
                raise Exception(f"Faltan columnas en tabla Kiosk: {missing}")
                
            logger.info("Estructura de tablas verificada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error verificando tablas: {str(e)}")
            raise

    @staticmethod
    def create_kiosk(name, store_name=None, location=None, latitude=None, longitude=None, owner_id=None):
        """
        Crea un nuevo kiosk con validaciones básicas.
        """
        if not name:
            raise ValueError("El nombre del kiosk es obligatorio")

        if (latitude is not None and longitude is None) or (latitude is None and longitude is not None):
            raise ValueError("Si se proporciona una coordenada, ambas son requeridas")

        kiosk = Kiosk(
            name=name, 
            store_name=store_name,
            location=location, 
            latitude=latitude,
            longitude=longitude,
            owner_id=owner_id,
            status='inactive',
            uuid=str(uuid.uuid4())
        )
        
        db.session.add(kiosk)
        db.session.commit()
        return kiosk

    @staticmethod
    def register_kiosk_from_spawner(serial, name, location, hardware_info):
        """
        Registra un kiosk inicializado por el spawner.
        Si ya existe, actualiza su información.
        """
        try:
            kiosk = Kiosk.query.filter_by(uuid=serial).first()
            
            if kiosk:
                # Actualizar kiosk existente
                kiosk.name = name
                kiosk.location = location
                kiosk.status = 'active'
                kiosk.last_online = datetime.utcnow()
            else:
                # Crear nuevo kiosk
                kiosk = Kiosk(
                    uuid=serial,
                    name=name,
                    location=location,
                    status='active',
                    last_online=datetime.utcnow()
                )
                db.session.add(kiosk)
            
            # Actualizar hardware info
            if hardware_info:
                kiosk.cpu_model = hardware_info.get('cpu_model')
                kiosk.ram_total = hardware_info.get('ram_total')
                kiosk.storage_total = hardware_info.get('storage_total')
                kiosk.ip_address = hardware_info.get('ip_address')
                kiosk.mac_address = hardware_info.get('mac_address')
            
            db.session.commit()
            logger.info(f"Kiosk registrado/actualizado: {name} ({serial})")
            return kiosk
            
        except Exception as e:
            logger.error(f"Error registrando kiosk desde spawner: {str(e)}")
            db.session.rollback()
            raise

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

    @staticmethod
    def get_nearby_kiosks(lat, lon, radius=5.0):
        """
        Obtiene kiosks cercanos a una ubicación dada.
        Args:
            lat (float): Latitud del punto central
            lon (float): Longitud del punto central
            radius (float): Radio de búsqueda en kilómetros
        Returns:
            List[Kiosk]: Lista de kiosks dentro del radio especificado
        """
        # Convertir radio de km a grados (aproximación)
        # 1 grado ≈ 111.32 km en el ecuador
        degree_radius = radius / 111.32
        
        # Buscar kiosks dentro del radio usando una aproximación rectangular
        # Esto es una simplificación y puede ser mejorada usando fórmulas más precisas
        nearby_kiosks = Kiosk.query.filter(
            Kiosk.latitude.between(lat - degree_radius, lat + degree_radius),
            Kiosk.longitude.between(lon - degree_radius, lon + degree_radius)
        ).all()
        
        # Filtrar resultados usando la fórmula de Haversine para mayor precisión
        result = []
        for kiosk in nearby_kiosks:
            distance = KioskService._calculate_distance(lat, lon, kiosk.latitude, kiosk.longitude)
            if distance <= radius:
                result.append(kiosk)
        
        return result

    @staticmethod
    def _calculate_distance(lat1, lon1, lat2, lon2):
        """
        Calcula la distancia entre dos puntos usando la fórmula de Haversine.
        Returns:
            float: Distancia en kilómetros
        """
        R = 6371  # Radio de la Tierra en km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance 

    @staticmethod
    def validate_kiosk_credentials(serial: str, credentials: dict) -> bool:
        """
        Valida las credenciales de un kiosk.
        Args:
            serial (str): UUID/Serial del kiosk
            credentials (dict): Credenciales a validar
        Returns:
            bool: True si las credenciales son válidas
        """
        kiosk = Kiosk.query.filter_by(uuid=serial).first()
        if not kiosk:
            return False
        return kiosk.validate_credentials(credentials)

    @staticmethod
    def is_registered(serial: str) -> bool:
        """
        Verifica si un kiosk está registrado en el sistema.
        Args:
            serial (str): UUID/Serial del kiosk
        Returns:
            bool: True si el kiosk está registrado
        """
        return Kiosk.query.filter_by(uuid=serial).first() is not None

    @staticmethod
    def update_kiosk_capabilities(kiosk_id: int, capabilities: dict) -> None:
        """
        Actualiza las capacidades de un kiosk.
        Args:
            kiosk_id (int): ID del kiosk
            capabilities (dict): Diccionario con las capacidades del kiosk
        """
        kiosk = KioskService.get_kiosk_by_id(kiosk_id)
        if kiosk:
            kiosk.capabilities = capabilities
            db.session.commit()

    @staticmethod
    def get_kiosk_by_id(kiosk_id):
        """
        Obtiene un kiosk por su ID.
        Args:
            kiosk_id (int): ID del kiosk a buscar
        Returns:
            Kiosk: El kiosk encontrado o None si no existe
        """
        return Kiosk.query.get(kiosk_id)

    @staticmethod
    def update_kiosk(kiosk_id, data):
        """
        Actualiza un kiosk existente.
        Args:
            kiosk_id (int): ID del kiosk a actualizar
            data (dict): Diccionario con los datos a actualizar
        Returns:
            Kiosk: El kiosk actualizado
        Raises:
            ValueError: Si el kiosk no existe o los datos son inválidos
        """
        kiosk = KioskService.get_kiosk_by_id(kiosk_id)
        if not kiosk:
            raise ValueError(f"Kiosk con ID {kiosk_id} no encontrado")

        # Validar nombre
        if 'name' in data and not data['name']:
            raise ValueError("El nombre del kiosk es obligatorio")

        # Actualizar campos básicos
        for field in ['name', 'location', 'status', 'cpu_model', 'ram_total', 
                     'storage_total', 'ip_address', 'mac_address', 'latitude',
                     'longitude', 'altitude']:
            if field in data:
                setattr(kiosk, field, data[field])

        db.session.commit()
        return kiosk 

    @staticmethod
    def delete_kiosk(kiosk_uuid):
        """
        Elimina un kiosk por su UUID.
        """
        kiosk = Kiosk.query.filter_by(uuid=kiosk_uuid).first()
        if not kiosk:
            raise ValueError(f"Kiosk con UUID {kiosk_uuid} no encontrado")
        
        # Eliminar datos de sensores asociados
        SensorData.query.filter_by(kiosk_id=kiosk.id).delete()
        
        db.session.delete(kiosk)
        db.session.commit()

        logger.info(f"Kiosk {kiosk_uuid} eliminado por el usuario {current_user.username}")
        
        return True 