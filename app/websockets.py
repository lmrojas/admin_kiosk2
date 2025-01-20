# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

"""
Módulo de WebSockets para comunicación en tiempo real.

Eventos del Cliente:
- connect: Conexión de cliente
- disconnect: Desconexión de cliente

Eventos del Servidor:
- metrics_update: Métricas del sistema (30s)
- new_alert: Nueva alerta de kiosk
- new_anomaly: Anomalía detectada
"""

from flask_socketio import SocketIO, emit
from flask import current_app
import json
from datetime import datetime
from app.models.kiosk import Kiosk, SensorData
from app.services.kiosk_service import KioskService
from app.services.kiosk_ai_service import KioskAIService

# Inicializar SocketIO
socketio = SocketIO()

def init_websockets(socketio_instance):
    """
    Inicializa la configuración de WebSockets
    
    Args:
        socketio_instance: Instancia de SocketIO a configurar
    """
    global socketio
    socketio = socketio_instance
    
    # Registrar handlers de eventos
    socketio.on_event('connect', handle_connect)
    socketio.on_event('disconnect', handle_disconnect)
    
    return socketio

def format_timestamp(dt):
    """Formatea timestamp para enviar al cliente"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')

@socketio.on('connect')
def handle_connect():
    """Manejador de conexión de cliente"""
    current_app.logger.info('Cliente WebSocket conectado')
    # Enviar datos iniciales
    emit_current_metrics()

@socketio.on('disconnect')
def handle_disconnect():
    """Manejador de desconexión de cliente"""
    current_app.logger.info('Cliente WebSocket desconectado')

def emit_current_metrics():
    """Emite métricas actuales a todos los clientes"""
    try:
        # Obtener métricas generales
        metrics = {
            'total_kiosks': Kiosk.query.count(),
            'online_kiosks': Kiosk.query.filter_by(status='active').count(),
            'kiosks_with_alerts': KioskService.count_kiosks_with_alerts(),
            'kiosks_with_anomalies': KioskService.count_kiosks_with_anomalies(),
            'timestamp': format_timestamp(datetime.utcnow())
        }

        # Calcular promedios de CPU y memoria
        recent_data = SensorData.query.order_by(
            SensorData.timestamp.desc()
        ).limit(100).all()

        if recent_data:
            metrics.update({
                'avg_cpu': sum(d.cpu_usage for d in recent_data) / len(recent_data),
                'avg_memory': sum(d.memory_usage for d in recent_data) / len(recent_data)
            })
        else:
            metrics.update({
                'avg_cpu': 0,
                'avg_memory': 0
            })

        socketio.emit('metrics_update', metrics)
    except Exception as e:
        current_app.logger.error(f'Error emitiendo métricas: {str(e)}')

def emit_new_alert(kiosk_id, message):
    """
    Emite una nueva alerta a todos los clientes
    
    Args:
        kiosk_id (int): ID del kiosk
        message (str): Mensaje de alerta
    """
    try:
        kiosk = Kiosk.query.get(kiosk_id)
        if kiosk:
            alert = {
                'kiosk_id': kiosk_id,
                'kiosk_name': kiosk.name,
                'message': message,
                'timestamp': format_timestamp(datetime.utcnow())
            }
            socketio.emit('new_alert', alert)
    except Exception as e:
        current_app.logger.error(f'Error emitiendo alerta: {str(e)}')

def emit_anomaly_detected(kiosk_id, probability, metrics):
    """
    Emite una detección de anomalía a todos los clientes
    
    Args:
        kiosk_id (int): ID del kiosk
        probability (float): Probabilidad de anomalía
        metrics (dict): Métricas que causaron la anomalía
    """
    try:
        kiosk = Kiosk.query.get(kiosk_id)
        if kiosk:
            anomaly = {
                'kiosk_id': kiosk_id,
                'kiosk_name': kiosk.name,
                'probability': probability,
                'description': f'Anomalía detectada en {kiosk.name} (Prob: {probability:.2f})',
                'metrics': metrics,
                'timestamp': format_timestamp(datetime.utcnow())
            }
            socketio.emit('new_anomaly', anomaly)
    except Exception as e:
        current_app.logger.error(f'Error emitiendo anomalía: {str(e)}')

def start_metrics_emission(app):
    """
    Inicia la emisión periódica de métricas
    
    Args:
        app: Aplicación Flask
    """
    def emit_metrics_job():
        with app.app_context():
            emit_current_metrics()
    
    # Programar emisión cada 30 segundos
    socketio.start_background_task(
        target=lambda: socketio.sleep(30),
        callback=emit_metrics_job
    ) 