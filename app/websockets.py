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

from flask import current_app
from flask_socketio import SocketIO
from datetime import datetime
from app.models.kiosk import Kiosk, SensorData
import logging

# Instancia global de SocketIO con manejo de errores mejorado
socketio = SocketIO(logger=True, engineio_logger=True)

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