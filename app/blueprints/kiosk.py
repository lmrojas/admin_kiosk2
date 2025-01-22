# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from flask import Blueprint, render_template, jsonify, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app.models.kiosk import Kiosk, KioskLocationHistory
from app.models.user import UserPermission
from app.services.kiosk_service import KioskService
from app.utils.decorators import permission_required
import logging
import uuid
from datetime import datetime

kiosk_bp = Blueprint('kiosk', __name__, url_prefix='/kiosk')
kiosk_service = KioskService()

@kiosk_bp.route('/')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def list_kiosks():
    """
    Lista todos los kiosks.
    Requiere permiso: VIEW_KIOSK
    """
    kiosks = kiosk_service.get_all_kiosks()
    return render_template('kiosk/list.html', kiosks=kiosks)

@kiosk_bp.route('/<int:kiosk_id>')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def view_kiosk(kiosk_id):
    """
    Ver detalles de un kiosk específico.
    Requiere permiso: VIEW_KIOSK
    """
    kiosk = kiosk_service.get_kiosk_by_id(kiosk_id)
    if not kiosk:
        flash('Kiosk no encontrado', 'error')
        return redirect(url_for('kiosk.list_kiosks'))
    
    return render_template('kiosk/details.html', kiosk=kiosk)

@kiosk_bp.route('/<int:kiosk_id>/update', methods=['GET', 'POST'])
@login_required
@permission_required(UserPermission.UPDATE_KIOSK.value)
def update_kiosk(kiosk_id):
    """
    Actualizar un kiosk existente.
    Requiere permiso: UPDATE_KIOSK
    """
    kiosk = kiosk_service.get_kiosk_by_id(kiosk_id)
    if not kiosk:
        flash('Kiosk no encontrado', 'error')
        return redirect(url_for('kiosk.list_kiosks'))
    
    if request.method == 'POST':
        try:
            data = {
                'name': request.form.get('name'),
                'location': request.form.get('location'),
                'status': request.form.get('status'),
                'cpu_model': request.form.get('cpu_model'),
                'ram_total': float(request.form.get('ram_total')) if request.form.get('ram_total') else None,
                'storage_total': float(request.form.get('storage_total')) if request.form.get('storage_total') else None,
                'ip_address': request.form.get('ip_address'),
                'mac_address': request.form.get('mac_address'),
                'latitude': float(request.form.get('latitude')) if request.form.get('latitude') else None,
                'longitude': float(request.form.get('longitude')) if request.form.get('longitude') else None,
                'altitude': float(request.form.get('altitude')) if request.form.get('altitude') else None
            }
            
            kiosk = kiosk_service.update_kiosk(kiosk_id, data)
            flash('Kiosk actualizado exitosamente', 'success')
            return redirect(url_for('kiosk.view_kiosk', kiosk_id=kiosk.id))
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            flash('Error al actualizar el kiosk', 'error')
            logging.error(f'Error actualizando kiosk: {str(e)}')
    
    return render_template('kiosk/update.html', kiosk=kiosk)

@kiosk_bp.route('/create', methods=['GET', 'POST'])
@login_required
@permission_required(UserPermission.CREATE_KIOSK.value)
def create_kiosk():
    """
    Crear un nuevo kiosk.
    Requiere permiso: CREATE_KIOSK
    """
    if request.method == 'POST':
        name = request.form.get('name')
        store_name = request.form.get('store_name')
        location = request.form.get('location')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        
        try:
            kiosk = kiosk_service.create_kiosk(
                name=name,
                store_name=store_name,
                location=location,
                latitude=float(latitude) if latitude else None,
                longitude=float(longitude) if longitude else None,
                owner_id=current_user.id
            )
            flash('Kiosk creado exitosamente', 'success')
            return redirect(url_for('kiosk.view_kiosk', kiosk_id=kiosk.id))
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            flash('Error al crear el kiosk', 'error')
            logging.error(f'Error creando kiosk: {str(e)}')
    
    return render_template('kiosk/create.html', uuid=str(uuid.uuid4()))

@kiosk_bp.route('/api/nearby')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_nearby_kiosks():
    """
    Obtener kiosks cercanos a una ubicación.
    Requiere permiso: VIEW_KIOSK
    """
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius = float(request.args.get('radius', 5))  # Radio en kilómetros, default 5km
        
        nearby_kiosks = kiosk_service.get_nearby_kiosks(lat, lon, radius)
        return jsonify([k.to_dict() for k in nearby_kiosks])
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Parámetros inválidos'}), 400
    except Exception as e:
        logging.error(f"Error obteniendo kiosks cercanos: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_bp.route('/api/kiosks')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_all_kiosks():
    """
    Obtiene todos los kiosks con su información de geolocalización.
    Requiere permiso: VIEW_KIOSK
    """
    try:
        kiosks = kiosk_service.get_all_kiosks()
        return jsonify([{
            'id': k.id,
            'name': k.name,
            'latitude': k.latitude,
            'longitude': k.longitude,
            'reported_latitude': k.reported_latitude,
            'reported_longitude': k.reported_longitude,
            'status': k.status
        } for k in kiosks if k.latitude and k.longitude])
        
    except Exception as e:
        logging.error(f'Error al obtener todos los kiosks: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_bp.route('/api/kiosk/<int:kiosk_id>/metrics', methods=['POST'])
@login_required
@permission_required(UserPermission.UPDATE_KIOSK.value)
def update_kiosk_metrics(kiosk_id):
    """
    Actualiza las métricas de un kiosk.
    Requiere permiso: UPDATE_KIOSK
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400

        # Registrar datos de sensores
        sensor_data = kiosk_service.register_sensor_data(
            kiosk_id=kiosk_id,
            cpu_usage=data.get('cpu_usage', 0),
            memory_usage=data.get('memory_usage', 0),
            network_latency=data.get('network_latency')
        )

        # Actualizar estado y hardware si se proporcionan
        if 'status' in data or 'hardware_info' in data:
            kiosk_service.update_kiosk_status(
                kiosk_id=kiosk_id,
                status=data.get('status'),
                hardware_info=data.get('hardware_info')
            )

        # Actualizar ubicación si se proporciona
        if 'location' in data:
            location_data = data['location']
            kiosk_service.update_kiosk(kiosk_id, {
                'latitude': location_data.get('latitude'),
                'longitude': location_data.get('longitude'),
                'altitude': location_data.get('altitude')
            })

        return jsonify({'success': True}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Error actualizando métricas del kiosk {kiosk_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_bp.route('/<uuid:kiosk_uuid>/delete', methods=['POST'])
@login_required
@permission_required(UserPermission.DELETE_KIOSK)
def delete_kiosk(kiosk_uuid):
    """
    Eliminar un kiosk existente.
    Requiere permiso: DELETE_KIOSK
    """
    try:
        kiosk_service.delete_kiosk(str(kiosk_uuid))
        flash('Kiosk eliminado exitosamente', 'success')
    except ValueError as e:
        flash(str(e), 'error')
    except Exception as e:
        flash('Error al eliminar el kiosk', 'error')
        logging.error(f'Error eliminando kiosk: {str(e)}')
    
    return redirect(url_for('kiosk.list_kiosks'))

@kiosk_bp.route('/map')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def show_map():
    """Muestra el mapa con todos los kiosks."""
    return render_template('kiosk/map.html')

@kiosk_bp.route('/<int:kiosk_id>/location_history')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def location_history(kiosk_id):
    """
    Vista dedicada para el historial de ubicaciones de un kiosk.
    Requiere permiso: VIEW_KIOSK
    """
    kiosk = kiosk_service.get_kiosk_by_id(kiosk_id)
    if not kiosk:
        flash('Kiosk no encontrado', 'error')
        return redirect(url_for('kiosk.list_kiosks'))
    
    return render_template('kiosk/location_history.html', kiosk=kiosk)

@kiosk_bp.route('/api/kiosk/<int:kiosk_id>/location_history')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_location_history(kiosk_id):
    """
    Obtiene el historial de ubicaciones de un kiosk con filtros opcionales.
    Requiere permiso: VIEW_KIOSK
    
    Parámetros de query:
    - date_from: Fecha inicial (ISO format)
    - date_to: Fecha final (ISO format)
    - location_type: Tipo de ubicación ('assigned' o 'reported')
    """
    try:
        kiosk = kiosk_service.get_kiosk_by_id(kiosk_id)
        if not kiosk:
            return jsonify({'error': 'Kiosk no encontrado'}), 404
            
        # Construir query base
        query = kiosk.location_history
        
        # Aplicar filtros si existen
        if request.args.get('date_from'):
            date_from = datetime.fromisoformat(request.args.get('date_from'))
            query = query.filter(KioskLocationHistory.timestamp >= date_from)
            
        if request.args.get('date_to'):
            date_to = datetime.fromisoformat(request.args.get('date_to'))
            query = query.filter(KioskLocationHistory.timestamp <= date_to)
            
        if request.args.get('location_type'):
            query = query.filter(KioskLocationHistory.location_type == request.args.get('location_type'))
        
        # Ordenar por fecha descendente
        history = query.order_by(KioskLocationHistory.timestamp.desc()).all()
        
        return jsonify([{
            'id': h.id,
            'latitude': h.latitude,
            'longitude': h.longitude,
            'accuracy': h.accuracy,
            'timestamp': h.timestamp.isoformat(),
            'location_type': h.location_type,
            'previous_latitude': h.previous_latitude,
            'previous_longitude': h.previous_longitude,
            'change_reason': h.change_reason,
            'created_at': h.created_at.isoformat(),
            'created_by': h.created_by
        } for h in history])
        
    except ValueError as e:
        return jsonify({'error': f'Error en formato de fecha: {str(e)}'}), 400
    except Exception as e:
        logging.error(f'Error al obtener historial de ubicaciones: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500 