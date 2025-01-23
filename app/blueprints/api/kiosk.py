"""
Blueprint para rutas API relacionadas con Kiosks.
Sigue el patrón MVT, manejando solo endpoints API.
"""

from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from app.models.user import UserPermission
from app.services.kiosk_service import KioskService
from app.blueprints.auth import admin_required, permission_required
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
kiosk_api = Blueprint('kiosk_api', __name__, url_prefix='/v1/kiosks')
kiosk_service = KioskService()

@kiosk_api.route('')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def list_kiosks():
    """API: Lista todos los kiosks."""
    try:
        kiosks = kiosk_service.get_all_kiosks()
        return jsonify([kiosk.to_dict() for kiosk in kiosks])
    except Exception as e:
        logger.error(f'Error al listar kiosks: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_api.route('/<uuid>')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_kiosk(uuid):
    """API: Obtiene un kiosk por su UUID."""
    try:
        kiosk = kiosk_service.get_kiosk_by_uuid(uuid)
        if not kiosk:
            return jsonify({'error': 'Kiosk no encontrado'}), 404
        return jsonify(kiosk.to_dict())
    except Exception as e:
        logger.error(f'Error al obtener kiosk {uuid}: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_api.route('/<uuid>', methods=['PUT'])
@login_required
@permission_required(UserPermission.UPDATE_KIOSK.value)
def update_kiosk(uuid):
    """API: Actualiza un kiosk."""
    try:
        data = request.get_json()
        updated_kiosk = kiosk_service.update_kiosk_by_uuid(uuid, data)
        return jsonify(updated_kiosk.to_dict())
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f'Error al actualizar kiosk {uuid}: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_api.route('/<uuid>', methods=['DELETE'])
@login_required
@permission_required(UserPermission.DELETE_KIOSK.value)
def delete_kiosk(uuid):
    """API: Elimina un kiosk."""
    try:
        kiosk_service.delete_kiosk(uuid)
        return jsonify({'message': 'Kiosk eliminado correctamente'})
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f'Error al eliminar kiosk {uuid}: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_api.route('/<uuid>/location')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_location_history(uuid):
    """API: Obtiene el historial de ubicaciones de un kiosk."""
    try:
        # Obtener parámetros de filtro
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        location_type = request.args.get('location_type')
        
        history = KioskService.get_location_history(
            uuid,
            date_from=datetime.fromisoformat(date_from) if date_from else None,
            date_to=datetime.fromisoformat(date_to) if date_to else None,
            location_type=location_type
        )
        
        return jsonify(history)
    except ValueError as e:
        return jsonify({'error': f'Error en formato de fecha: {str(e)}'}), 400
    except Exception as e:
        logger.error(f'Error al obtener historial de ubicaciones para kiosk {uuid}: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_api.route('/nearby')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_nearby_kiosks():
    """API: Obtener kiosks cercanos a una ubicación."""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius = float(request.args.get('radius', 5))  # Radio en kilómetros, default 5km
        
        nearby_kiosks = kiosk_service.get_nearby_kiosks(lat, lon, radius)
        return jsonify([k.to_dict() for k in nearby_kiosks])
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Parámetros inválidos'}), 400
    except Exception as e:
        logger.error(f"Error obteniendo kiosks cercanos: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_api.route('/<uuid>/telemetry', methods=['POST'])
@login_required
@permission_required(UserPermission.UPDATE_KIOSK.value)
def update_telemetry(uuid):
    """API: Actualiza la telemetría de un kiosk."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos de telemetría'}), 400
            
        result = kiosk_service.update_telemetry(uuid, data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f'Error al actualizar telemetría del kiosk {uuid}: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_api.route('/<uuid>/status')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_status(uuid):
    """API: Obtiene el estado actual de un kiosk."""
    try:
        status = kiosk_service.get_kiosk_status(uuid)
        if not status:
            return jsonify({'error': 'Kiosk no encontrado'}), 404
        return jsonify(status)
    except Exception as e:
        logger.error(f'Error al obtener estado del kiosk {uuid}: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500 