"""
Servicio para manejar eventos WebSocket.
Sigue el patrón MVT + S, centralizando la lógica de WebSocket.
"""

from flask_socketio import Namespace, emit
from flask import current_app
from app.models.kiosk import Kiosk, SensorData
import logging
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

class KioskNamespace(Namespace):
    """Namespace para eventos de kiosks."""
    
    def __init__(self, namespace=None):
        super().__init__(namespace)
    
    def on_connect(self):
        """Maneja conexión de un kiosk."""
        try:
            logger.info('Nueva conexión WebSocket establecida')
            emit('connection_established', {'status': 'connected'})
        except Exception as e:
            logger.error(f'Error en conexión WebSocket: {str(e)}')
            emit('error', {'error': 'Error al establecer conexión'})
    
    def on_disconnect(self):
        """Maneja desconexión de un kiosk."""
        try:
            logger.info('Cliente WebSocket desconectado')
        except Exception as e:
            logger.error(f'Error en desconexión WebSocket: {str(e)}')
    
    def on_registration(self, data):
        """
        Maneja registro de un kiosk.
        Redirige al endpoint API para mantener un único punto de entrada.
        """
        try:
            if not data or 'kiosk_uuid' not in data:
                logger.error('Datos de registro inválidos')
                emit('registration_error', {'error': 'Datos de registro inválidos'})
                return
            
            # Usar el endpoint API existente
            response = requests.post(
                f"{current_app.config['API_URL']}/api/v1/kiosks",
                json=data
            )
            
            if response.status_code == 201:
                result = response.json()
                emit('registration_success', result)
                logger.info(f'Kiosk registrado exitosamente: {data["kiosk_uuid"]}')
            else:
                error = response.json().get('error', 'Error desconocido')
                emit('registration_error', {'error': error})
                logger.error(f'Error en registro de kiosk: {error}')
                
        except Exception as e:
            logger.error(f'Error en registro de kiosk: {str(e)}')
            emit('registration_error', {'error': str(e)})
    
    def on_telemetry(self, data):
        """Maneja datos de telemetría de un kiosk."""
        try:
            if not data or 'kiosk_uuid' not in data:
                logger.error('Datos de telemetría inválidos')
                emit('telemetry_error', {'error': 'Datos de telemetría inválidos'})
                return
            
            # Usar el endpoint API existente
            response = requests.post(
                f"{current_app.config['API_URL']}/api/v1/kiosks/{data['kiosk_uuid']}/telemetry",
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                emit('telemetry_processed', result)
                logger.info(f'Telemetría procesada para kiosk: {data["kiosk_uuid"]}')
            else:
                error = response.json().get('error', 'Error desconocido')
                emit('telemetry_error', {'error': error})
                logger.error(f'Error procesando telemetría: {error}')
                
        except Exception as e:
            logger.error(f'Error procesando telemetría: {str(e)}')
            emit('telemetry_error', {'error': str(e)})

class WebSocketService:
    """Servicio para manejar WebSockets."""
    
    @staticmethod
    def init_websockets():
        """Inicializa los WebSockets."""
        from app import socketio
        socketio.on_namespace(KioskNamespace('/kiosk'))

    @staticmethod
    def emit_current_metrics():
        """Emite las métricas actuales de todos los kioscos."""
        try:
            from app import socketio
            total_kiosks = Kiosk.query.count()
            online_kiosks = Kiosk.query.filter_by(is_online=True).count()
            
            recent_data = SensorData.query.order_by(
                SensorData.timestamp.desc()
            ).limit(10).all()
            
            if recent_data:
                avg_cpu = sum(data.cpu_usage for data in recent_data) / len(recent_data)
                avg_memory = sum(data.memory_usage for data in recent_data) / len(recent_data)
            else:
                avg_cpu = avg_memory = 0
                
            socketio.emit('metrics_update', {
                'total_kiosks': total_kiosks,
                'online_kiosks': online_kiosks,
                'avg_cpu': avg_cpu,
                'avg_memory': avg_memory,
                'timestamp': WebSocketService.format_timestamp(datetime.now())
            })
            
        except Exception as e:
            logger.error(f'Error al emitir métricas: {str(e)}')

    @staticmethod
    def format_timestamp(dt: datetime) -> str:
        """Formatea un objeto datetime a string."""
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def emit_new_alert(kiosk_id: int, message: str) -> None:
        """Emite una nueva alerta para un kiosk específico."""
        try:
            from app import socketio
            kiosk = Kiosk.query.get(kiosk_id)
            if not kiosk:
                current_app.logger.warning(f'Intento de emitir alerta para kiosk inexistente ID={kiosk_id}')
                return
                
            socketio.emit('new_alert', {
                'kiosk_id': kiosk_id,
                'kiosk_name': kiosk.name,
                'message': message,
                'timestamp': WebSocketService.format_timestamp(datetime.now())
            })
            
        except Exception as e:
            current_app.logger.error(f'Error al emitir alerta: {str(e)}')

    @staticmethod
    def emit_anomaly_detected(kiosk_id: int, probability: float, metrics: dict) -> None:
        """Emite una detección de anomalía para un kiosk específico."""
        try:
            from app import socketio
            kiosk = Kiosk.query.get(kiosk_id)
            if not kiosk:
                current_app.logger.warning(f'Intento de emitir anomalía para kiosk inexistente ID={kiosk_id}')
                return
                
            socketio.emit('new_anomaly', {
                'kiosk_id': kiosk_id,
                'kiosk_name': kiosk.name,
                'probability': probability,
                'metrics': metrics,
                'timestamp': WebSocketService.format_timestamp(datetime.now())
            })
            
        except Exception as e:
            current_app.logger.error(f'Error al emitir anomalía: {str(e)}')

websocket_service = WebSocketService() 