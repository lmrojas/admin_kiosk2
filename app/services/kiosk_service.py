# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from app.models.base import db
from app.models.kiosk import Kiosk, SensorData
from app.services.kiosk_ai_service import KioskAIService
from datetime import datetime, timedelta
import uuid
import math
import logging
from typing import Dict, Optional, Tuple
from sqlalchemy.exc import SQLAlchemyError

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
    def update_kiosk_status(kiosk_id, status, hardware_info=None, system_info=None, 
                           security_info=None, time_info=None):
        """
        Actualiza el estado y toda la información de un kiosk.
        """
        kiosk = Kiosk.query.get(kiosk_id)
        if not kiosk:
            raise ValueError(f"Kiosk con ID {kiosk_id} no encontrado")

        # Actualizar estado básico
        kiosk.status = status
        kiosk.last_online = datetime.utcnow()

        # Actualizar información de hardware
        if hardware_info:
            kiosk.cpu_model = hardware_info.get('cpu_model', kiosk.cpu_model)
            kiosk.ram_total = hardware_info.get('ram_total', kiosk.ram_total)
            kiosk.storage_total = hardware_info.get('storage_total', kiosk.storage_total)
            kiosk.disk_usage = hardware_info.get('disk_usage', kiosk.disk_usage)
            kiosk.disk_free = hardware_info.get('disk_free', kiosk.disk_free)
            kiosk.fan_rpm = hardware_info.get('fan_rpm', kiosk.fan_rpm)
            kiosk.ip_address = hardware_info.get('ip_address', kiosk.ip_address)
            kiosk.public_ip = hardware_info.get('public_ip', kiosk.public_ip)
            kiosk.mac_address = hardware_info.get('mac_address', kiosk.mac_address)
            kiosk.wifi_signal_strength = hardware_info.get('wifi_signal_strength', kiosk.wifi_signal_strength)
            kiosk.connection_speed = hardware_info.get('connection_speed', kiosk.connection_speed)
            kiosk.packets_sent = hardware_info.get('packets_sent', kiosk.packets_sent)
            kiosk.packets_received = hardware_info.get('packets_received', kiosk.packets_received)
            kiosk.packets_lost = hardware_info.get('packets_lost', kiosk.packets_lost)

        # Actualizar información del sistema
        if system_info:
            kiosk.os_name = system_info.get('os_name', kiosk.os_name)
            kiosk.os_version = system_info.get('os_version', kiosk.os_version)
            kiosk.os_platform = system_info.get('os_platform', kiosk.os_platform)
            kiosk.chromium_status = system_info.get('chromium_status', kiosk.chromium_status)
            kiosk.chromium_version = system_info.get('chromium_version', kiosk.chromium_version)

        # Actualizar información de seguridad
        if security_info:
            if security_info.get('last_unauthorized_access'):
                kiosk.last_unauthorized_access = datetime.fromisoformat(security_info['last_unauthorized_access'])
            kiosk.block_reason = security_info.get('block_reason', kiosk.block_reason)
            kiosk.door_status = security_info.get('door_status', kiosk.door_status)

        # Actualizar información de tiempo
        if time_info:
            kiosk.local_timezone = time_info.get('local_timezone', kiosk.local_timezone)
            kiosk.utc_offset = time_info.get('utc_offset', kiosk.utc_offset)

        db.session.commit()
        return kiosk

    @staticmethod
    def register_sensor_data(kiosk_id, cpu_usage, memory_usage, network_latency=None, 
                           cpu_temperature=None, ambient_temperature=None, 
                           humidity=None, voltage=None):
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
            network_latency=network_latency,
            cpu_temperature=cpu_temperature,
            ambient_temperature=ambient_temperature,
            humidity=humidity,
            voltage=voltage
        )
        db.session.add(sensor_data)

        # Usar servicio de IA para predecir anomalías
        ai_service = KioskAIService()
        anomaly_prob = ai_service.predict_anomaly({
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'network_latency': network_latency or 0,
            'cpu_temperature': cpu_temperature or 0,
            'ambient_temperature': ambient_temperature or 0,
            'humidity': humidity or 0,
            'voltage': voltage or 0
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

    @staticmethod
    def validate_registration_data(data: Dict) -> Tuple[bool, str]:
        """Valida los datos de registro de un kiosk."""
        if not data.get('kiosk_uuid'):
            return False, 'UUID es requerido'
        return True, ''

    @staticmethod
    def validate_telemetry_data(data: Dict) -> Tuple[bool, str, list]:
        """Valida los datos de telemetría."""
        required_sections = ['system_status', 'hardware', 'sensors', 'security', 'network']
        missing = [section for section in required_sections if section not in data]
        if missing:
            return False, f'Datos incompletos: faltan {", ".join(missing)}', missing
        return True, '', []

    @classmethod
    def handle_kiosk_connection(cls, kiosk_uuid: str, socket_id: str) -> Tuple[bool, str, Optional[Kiosk]]:
        """Maneja la conexión de un kiosk."""
        try:
            kiosk = Kiosk.query.filter_by(uuid=kiosk_uuid).first()
            if not kiosk:
                return False, 'Kiosk no registrado en el sistema', None
                
            if kiosk.is_online and kiosk.socket_id and kiosk.socket_id != socket_id:
                return False, 'Kiosk ya está conectado desde otra sesión', None
                
            kiosk.socket_id = socket_id
            kiosk.is_online = True
            kiosk.last_online = datetime.utcnow()
            db.session.commit()
            
            return True, 'Conexión exitosa', kiosk
            
        except SQLAlchemyError as e:
            db.session.rollback()
            return False, f'Error de base de datos: {str(e)}', None

    @classmethod
    def handle_kiosk_disconnection(cls, socket_id: str) -> None:
        """Maneja la desconexión de un kiosk."""
        try:
            kiosks = Kiosk.query.filter_by(socket_id=socket_id).all()
            for kiosk in kiosks:
                kiosk.is_online = False
                kiosk.socket_id = None
                kiosk.last_offline = datetime.utcnow()
            db.session.commit()
        except SQLAlchemyError as e:
            db.session.rollback()
            logging.error(f'Error en desconexión: {str(e)}')

    @classmethod
    def process_telemetry(cls, kiosk: Kiosk, telemetry_data: Dict) -> Tuple[bool, str, int]:
        """Procesa los datos de telemetría de un kiosk."""
        try:
            with db.session.begin_nested():
                # Extraer datos
                system_status = telemetry_data.get('system_status', {})
                hardware = telemetry_data.get('hardware', {})
                sensors = telemetry_data.get('sensors', {})
                security = telemetry_data.get('security', {})
                network = telemetry_data.get('network', {})
                
                # Actualizar estado
                cls.update_kiosk_status(
                    kiosk_id=kiosk.id,
                    status=system_status.get('current_status', 'unknown'),
                    hardware_info={
                        'cpu_model': hardware.get('cpu', {}).get('model'),
                        'ram_total': hardware.get('memory', {}).get('total'),
                        'disk_usage': hardware.get('disk_usage'),
                        'disk_free': hardware.get('disk_free'),
                        'fan_rpm': hardware.get('fan_rpm'),
                        'public_ip': network.get('public_ip'),
                        'wifi_signal_strength': network.get('signal_quality'),
                        'connection_speed': network.get('connection_speed'),
                        'packets_sent': network.get('packets', {}).get('sent'),
                        'packets_received': network.get('packets', {}).get('received'),
                        'packets_lost': network.get('packets', {}).get('lost')
                    },
                    system_info={
                        'os_name': system_status.get('os_name'),
                        'os_version': system_status.get('os_version'),
                        'os_platform': system_status.get('os_platform'),
                        'chromium_status': system_status.get('chromium_status'),
                        'chromium_version': system_status.get('chromium_version')
                    },
                    security_info={
                        'last_unauthorized_access': security.get('last_unauthorized_access'),
                        'block_reason': system_status.get('block_reason'),
                        'door_status': sensors.get('door')
                    },
                    time_info={
                        'local_timezone': system_status.get('timezone'),
                        'utc_offset': system_status.get('utc_offset')
                    }
                )
                
                # Registrar datos de sensores
                cls.register_sensor_data(
                    kiosk_id=kiosk.id,
                    cpu_usage=hardware.get('cpu', {}).get('usage', 0),
                    memory_usage=hardware.get('memory', {}).get('percent', 0),
                    network_latency=network.get('latency'),
                    cpu_temperature=hardware.get('cpu', {}).get('temperature'),
                    ambient_temperature=sensors.get('temperature'),
                    humidity=sensors.get('humidity'),
                    voltage=sensors.get('voltage')
                )
                
                # Actualizar timestamp
                kiosk.last_telemetry = datetime.utcnow()
                db.session.commit()
                
                return True, 'Telemetría procesada correctamente', 5  # 5 segundos para próximo envío
                
        except SQLAlchemyError as e:
            db.session.rollback()
            return False, f'Error de base de datos: {str(e)}', 30  # 30 segundos para reintentar
        except Exception as e:
            return False, f'Error interno: {str(e)}', 60  # 1 minuto para reintentar

    @classmethod
    def update_kiosk_status(cls, kiosk_id: int, status: str, hardware_info: Dict, 
                           system_info: Dict, security_info: Dict, time_info: Dict) -> None:
        """Actualiza el estado de un kiosk."""
        try:
            kiosk = Kiosk.query.get(kiosk_id)
            if not kiosk:
                raise ValueError(f'Kiosk {kiosk_id} no encontrado')
            
            # Actualizar información de hardware
            for key, value in hardware_info.items():
                if hasattr(kiosk, key):
                    setattr(kiosk, key, value)
                    
            # Actualizar información de sistema
            for key, value in system_info.items():
                if hasattr(kiosk, key):
                    setattr(kiosk, key, value)
                    
            # Actualizar información de seguridad
            for key, value in security_info.items():
                if hasattr(kiosk, key):
                    setattr(kiosk, key, value)
                    
            # Actualizar información de tiempo
            for key, value in time_info.items():
                if hasattr(kiosk, key):
                    setattr(kiosk, key, value)
                    
            kiosk.status = status
            db.session.commit()
            
        except SQLAlchemyError as e:
            db.session.rollback()
            raise

    @classmethod
    def register_sensor_data(cls, kiosk_id: int, cpu_usage: float, memory_usage: float,
                           network_latency: Optional[float] = None, cpu_temperature: Optional[float] = None,
                           ambient_temperature: Optional[float] = None, humidity: Optional[float] = None,
                           voltage: Optional[float] = None) -> None:
        """Registra datos de sensores de un kiosk."""
        try:
            sensor_data = SensorData(
                kiosk_id=kiosk_id,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                network_latency=network_latency,
                cpu_temperature=cpu_temperature,
                ambient_temperature=ambient_temperature,
                humidity=humidity,
                voltage=voltage
            )
            db.session.add(sensor_data)
            db.session.commit()
            
            # Actualizar métricas de IA
            kiosk = Kiosk.query.get(kiosk_id)
            if kiosk:
                ai_service = KioskAIService()
                anomaly_prob = ai_service.detect_anomalies(kiosk)
                kiosk.update_health_metrics(anomaly_prob)
                db.session.commit()
                
        except SQLAlchemyError as e:
            db.session.rollback()
            raise 

    @staticmethod
    def get_location_history(uuid, date_from=None, date_to=None, location_type=None):
        """
        Obtiene el historial de ubicaciones de un kiosk.
        
        Args:
            uuid (str): UUID del kiosk
            date_from (datetime, optional): Fecha inicial del filtro
            date_to (datetime, optional): Fecha final del filtro
            location_type (str, optional): Tipo de ubicación ('assigned' o 'reported')
            
        Returns:
            list: Lista de registros de ubicación
            
        Raises:
            ValueError: Si el kiosk no existe
        """
        kiosk = Kiosk.query.filter_by(uuid=uuid).first()
        if not kiosk:
            raise ValueError(f"Kiosk con UUID {uuid} no encontrado")
            
        # Por ahora solo retornamos la ubicación actual
        # TODO: Implementar historial completo cuando se agregue la tabla de historial
        location = {
            'timestamp': kiosk.last_online or datetime.utcnow(),
            'latitude': kiosk.latitude,
            'longitude': kiosk.longitude,
            'reported_latitude': kiosk.reported_latitude,
            'reported_longitude': kiosk.reported_longitude,
            'location_type': location_type or 'assigned'
        }
        
        return [location] 