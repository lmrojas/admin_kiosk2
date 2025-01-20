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
    
    return render_template('kiosk/detail.html', kiosk=kiosk)

@kiosk_bp.route('/<int:kiosk_id>/metrics')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_kiosk_metrics(kiosk_id):
    """
    Obtener métricas de un kiosk.
    Requiere permiso: VIEW_KIOSK
    """
    metrics = kiosk_service.get_kiosk_metrics(kiosk_id)
    return jsonify(metrics)

@kiosk_bp.route('/<int:kiosk_id>/status')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def get_kiosk_status(kiosk_id):
    """
    Obtener estado actual de un kiosk.
    Requiere permiso: VIEW_KIOSK
    """
    status = kiosk_service.get_kiosk_status(kiosk_id)
    return jsonify(status)

@kiosk_bp.route('/<int:kiosk_id>/update', methods=['POST'])
@login_required
@permission_required(UserPermission.UPDATE_KIOSK.value)
def update_kiosk(kiosk_id):
    """
    Actualizar información de un kiosk.
    Requiere permiso: UPDATE_KIOSK
    """
    data = request.get_json()
    try:
        kiosk = kiosk_service.update_kiosk(kiosk_id, data)
        return jsonify({'message': 'Kiosk actualizado exitosamente', 'kiosk': kiosk.to_dict()})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Error actualizando kiosk {kiosk_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@kiosk_bp.route('/<int:kiosk_id>/location', methods=['POST'])
@login_required
@permission_required(UserPermission.UPDATE_KIOSK.value)
def update_kiosk_location(kiosk_id):
    """
    Actualizar ubicación de un kiosk.
    Requiere permiso: UPDATE_KIOSK
    """
    data = request.get_json()
    try:
        kiosk = kiosk_service.update_kiosk_location(
            kiosk_id,
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            altitude=data.get('altitude')
        )
        return jsonify({'message': 'Ubicación actualizada exitosamente', 'kiosk': kiosk.to_dict()})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Error actualizando ubicación del kiosk {kiosk_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

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