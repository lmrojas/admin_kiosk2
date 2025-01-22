# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

"""
Módulo de WebSockets para el Sistema de Kioscos Inteligentes

Este módulo maneja la comunicación en tiempo real entre el servidor y los clientes
a través de WebSocket, permitiendo:
1. Emisión de métricas actuales de los kioscos
2. Notificación de nuevas alertas
3. Detección y notificación de anomalías

La comunicación es bidireccional y asíncrona, siguiendo el patrón pub/sub
para mantener actualizados los dashboards y monitores en tiempo real.

Eventos emitidos:
- metrics_update: Actualización periódica de métricas
- new_alert: Notificación de nuevas alertas
- new_anomaly: Detección de comportamientos anómalos

Autor: Sistema Automatizado siguiendo @cura.md
Fecha: 2024
"""

from flask import current_app, request
from flask_socketio import SocketIO, emit
from datetime import datetime
from app.models.kiosk import Kiosk, SensorData
import logging
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError

# Instancia global de SocketIO con manejo de errores mejorado
socketio = SocketIO(logger=True, engineio_logger=True)
db = SQLAlchemy()

def init_websockets(socketio_instance: SocketIO) -> None:
    """
    Inicializa la configuración de WebSocket y registra los handlers.
    Esta función es llamada durante la inicialización de la aplicación.
    
    Args:
        socketio_instance (SocketIO): Instancia de SocketIO a configurar
    """
    try:
        global socketio
        socketio = socketio_instance
        
        # Configuración de logging segura
        logger = logging.getLogger('websockets')
        logger.info('WebSocket inicializado correctamente')
        
    except Exception as e:
        # Log seguro sin depender de current_app
        logger = logging.getLogger('websockets')
        logger.error(f'Error al inicializar WebSocket: {str(e)}')

def format_timestamp(dt: datetime) -> str:
    """
    Formatea un objeto datetime a string en formato ISO.
    
    Args:
        dt (datetime): Objeto datetime a formatear
        
    Returns:
        str: Timestamp formateado como 'YYYY-MM-DD HH:MM:SS'
    """
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def emit_current_metrics() -> None:
    """
    Emite las métricas actuales de todos los kioscos.
    Incluye:
    - Total de kioscos
    - Kioscos en línea
    - Promedio de uso de CPU
    - Promedio de uso de memoria
    
    Evento: 'metrics_update'
    """
    try:
        # Obtener conteos de kioscos
        total_kiosks = Kiosk.query.count()
        online_kiosks = Kiosk.query.filter_by(is_online=True).count()
        
        # Obtener últimas métricas de sensores
        recent_data = SensorData.query.order_by(
            SensorData.timestamp.desc()
        ).limit(10).all()
        
        # Calcular promedios
        if recent_data:
            avg_cpu = sum(data.cpu_usage for data in recent_data) / len(recent_data)
            avg_memory = sum(data.memory_usage for data in recent_data) / len(recent_data)
        else:
            avg_cpu = avg_memory = 0
            
        # Emitir métricas
        socketio.emit('metrics_update', {
            'total_kiosks': total_kiosks,
            'online_kiosks': online_kiosks,
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'timestamp': format_timestamp(datetime.now())
        })
        
    except Exception as e:
        current_app.logger.error(f'Error al emitir métricas: {str(e)}')

def emit_new_alert(kiosk_id: int, message: str) -> None:
    """
    Emite una nueva alerta para un kiosk específico.
    
    Args:
        kiosk_id (int): ID del kiosk que generó la alerta
        message (str): Mensaje de la alerta
        
    Evento: 'new_alert'
    """
    try:
        kiosk = Kiosk.query.get(kiosk_id)
        if not kiosk:
            current_app.logger.warning(f'Intento de emitir alerta para kiosk inexistente ID={kiosk_id}')
            return
            
        socketio.emit('new_alert', {
            'kiosk_id': kiosk_id,
            'kiosk_name': kiosk.name,
            'message': message,
            'timestamp': format_timestamp(datetime.now())
        })
        
    except Exception as e:
        current_app.logger.error(f'Error al emitir alerta: {str(e)}')

def emit_anomaly_detected(kiosk_id: int, probability: float, metrics: dict) -> None:
    """
    Emite una detección de anomalía para un kiosk específico.
    
    Args:
        kiosk_id (int): ID del kiosk donde se detectó la anomalía
        probability (float): Probabilidad de que sea una anomalía real (0-1)
        metrics (dict): Métricas que causaron la detección de anomalía
        
    Evento: 'new_anomaly'
    """
    try:
        kiosk = Kiosk.query.get(kiosk_id)
        if not kiosk:
            current_app.logger.warning(f'Intento de emitir anomalía para kiosk inexistente ID={kiosk_id}')
            return
            
        socketio.emit('new_anomaly', {
            'kiosk_id': kiosk_id,
            'kiosk_name': kiosk.name,
            'probability': probability,
            'metrics': metrics,
            'timestamp': format_timestamp(datetime.now())
        })
        
    except Exception as e:
        current_app.logger.error(f'Error al emitir anomalía: {str(e)}')

@socketio.on('connect')
def handle_connect():
    """Maneja conexiones nuevas de kiosks"""
    logger = logging.getLogger('websockets')
    logger.info(f'Nueva conexión WebSocket desde {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    """Maneja desconexiones de kiosks"""
    logger = logging.getLogger('websockets')
    try:
        # Marcar kiosks como offline al desconectar
        kiosks = Kiosk.query.filter_by(socket_id=request.sid).all()
        for kiosk in kiosks:
            kiosk.is_online = False
            kiosk.socket_id = None
        db.session.commit()
        emit('kiosk_status_change', broadcast=True)
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f'Error actualizando estado de kiosks: {str(e)}')
    except Exception as e:
        logger.error(f'Error en desconexión: {str(e)}')
    finally:
        logger.info(f'Desconexión WebSocket de {request.sid}')

@socketio.on('registration')
def handle_registration(data):
    """Maneja el registro inicial de un kiosk"""
    logger = logging.getLogger('websockets')
    try:
        serial = data.get('serial')
        name = data.get('name')
        
        logger.info(f'Intento de registro: {name} ({serial})')
        
        # Verificar registro en BD dentro de una transacción
        kiosk = Kiosk.query.filter_by(uuid=serial).first()
        if kiosk:
            kiosk.is_online = True
            kiosk.last_online = datetime.now()
            kiosk.socket_id = request.sid
            db.session.commit()
            
            # Notificar cambio de estado
            emit('kiosk_status_change', broadcast=True)
            
            logger.info(f'Kiosk registrado: {name} ({serial})')
            return {'status': 'registered'}
        else:
            logger.warning(f'Intento de registro de kiosk no autorizado: {serial}')
            return {'status': 'unauthorized'}
            
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f'Error de base de datos en registro: {str(e)}')
        return {'status': 'error', 'message': 'Database error'}
    except Exception as e:
        logger.error(f'Error en registro de kiosk: {str(e)}')
        return {'status': 'error', 'message': 'Internal error'}

@socketio.on('kiosk_update')
def handle_kiosk_update(data):
    """Maneja actualizaciones de datos de kiosks"""
    logger = logging.getLogger('websockets')
    try:
        serial = data.get('serial')
        kiosk = Kiosk.query.filter_by(uuid=serial).first()
        
        if kiosk and kiosk.socket_id == request.sid:
            # Actualizar datos del kiosk dentro de una transacción
            kiosk.last_online = datetime.now()
            kiosk.status = data.get('status', {}).get('current', 'unknown')
            kiosk.is_online = True
            
            # Extraer valores de sensores con manejo de errores
            sensors = data.get('sensors', {})
            sensor_data = SensorData(
                kiosk_id=kiosk.id,
                temperature=float(sensors.get('temperature', {}).get('value', 0)),
                humidity=float(sensors.get('humidity', {}).get('value', 0)),
                door_status=sensors.get('door_status', 'unknown'),
                printer_status=sensors.get('printer_status', 'unknown'),
                network_quality=float(sensors.get('network_quality', {}).get('value', 0)),
                voltage=float(sensors.get('voltage', 220)),
                ventilation=sensors.get('ventilation', 'normal')
            )
            db.session.add(sensor_data)
            db.session.commit()
            
            # Emitir actualización a todos los clientes
            emit('kiosk_data_update', {
                'kiosk_id': kiosk.id,
                'status': kiosk.status,
                'last_update': kiosk.last_online.isoformat()
            }, broadcast=True)
            
            logger.info(f'Datos actualizados para kiosk {serial}')
            
        else:
            logger.warning(f'Actualización rechazada para kiosk {serial}: no registrado o sesión inválida')
            
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f'Error de base de datos en actualización: {str(e)}')
    except Exception as e:
        logger.error(f'Error procesando actualización de kiosk: {str(e)}') 