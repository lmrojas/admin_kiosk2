# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from flask import Blueprint, render_template, jsonify, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app.models.kiosk import Kiosk
from app.models.user import UserPermission
from app.services.kiosk_service import KioskService
from app.utils.decorators import permission_required
import logging

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
        location = request.form.get('location')
        
        try:
            kiosk = kiosk_service.create_kiosk(
                name=name,
                location=location,
                owner_id=current_user.id
            )
            flash('Kiosk creado exitosamente', 'success')
            return redirect(url_for('kiosk.view_kiosk', kiosk_id=kiosk.id))
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            flash('Error al crear el kiosk', 'error')
            logging.error(f'Error creando kiosk: {str(e)}')
    
    return render_template('kiosk/create.html')

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