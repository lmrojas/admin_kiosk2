from flask import Blueprint, jsonify, request, current_app, render_template
from flask_jwt_extended import jwt_required
from app.services.geolocation_service import GeolocationService
from app.utils.decorators import permission_required
from app.models.kiosk import Kiosk

kiosk_bp = Blueprint('kiosk', __name__)

@kiosk_bp.route('/api/nearby')
@jwt_required()
@permission_required('kiosk_view')
def get_nearby_kiosks():
    """Obtiene los kiosks cercanos a una ubicación dada."""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius = float(request.args.get('radius', 5.0))  # Radio en km, default 5km
        
        geolocation_service = GeolocationService()
        nearby_kiosks = geolocation_service.get_nearby_kiosks(lat, lon, radius)
        
        return jsonify([{
            'id': k.id,
            'name': k.name,
            'latitude': k.latitude,
            'longitude': k.longitude,
            'status': k.status
        } for k in nearby_kiosks])
        
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Parámetros inválidos'}), 400
    except Exception as e:
        current_app.logger.error(f'Error al obtener kiosks cercanos: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500 

@kiosk_bp.route('/api/all')
@jwt_required()
@permission_required('kiosk_view')
def get_all_kiosks():
    """Obtiene todos los kiosks con su información de geolocalización."""
    try:
        kiosks = Kiosk.query.all()
        return jsonify([{
            'id': k.id,
            'name': k.name,
            'latitude': k.latitude,
            'longitude': k.longitude,
            'status': k.status
        } for k in kiosks if k.latitude and k.longitude])
        
    except Exception as e:
        current_app.logger.error(f'Error al obtener todos los kiosks: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500 

@kiosk_bp.route('/map')
@jwt_required()
@permission_required('kiosk_view')
def show_map():
    """Muestra el mapa con todos los kiosks."""
    return render_template('kiosk/map.html') 