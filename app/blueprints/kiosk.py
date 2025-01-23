"""
Blueprint para vistas web relacionadas con Kiosks.
Sigue el patrón MVT, manejando solo vistas web.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app.models.user import UserPermission
from app.services.kiosk_service import KioskService
from app.blueprints.auth import admin_required, permission_required
import logging
import uuid

logger = logging.getLogger(__name__)
kiosk_bp = Blueprint('kiosk', __name__, url_prefix='/kiosk')
kiosk_service = KioskService()

@kiosk_bp.route('/')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def list_kiosks():
    """Lista todos los kiosks en vista web."""
    kiosks = kiosk_service.get_all_kiosks()
    return render_template('kiosk/list.html', kiosks=kiosks)

@kiosk_bp.route('/<int:kiosk_id>')
@login_required
@permission_required(UserPermission.VIEW_KIOSK.value)
def view_kiosk(kiosk_id):
    """Ver detalles de un kiosk específico."""
    kiosk = kiosk_service.get_kiosk_by_id(kiosk_id)
    if not kiosk:
        flash('Kiosk no encontrado', 'error')
        return redirect(url_for('kiosk.list_kiosks'))
    
    return render_template('kiosk/details.html', kiosk=kiosk)

@kiosk_bp.route('/create', methods=['GET', 'POST'])
@login_required
@permission_required(UserPermission.CREATE_KIOSK.value)
def create_kiosk():
    """Crear un nuevo kiosk."""
    if request.method == 'POST':
        try:
            kiosk = kiosk_service.create_kiosk(
                name=request.form.get('name'),
                store_name=request.form.get('store_name'),
                location=request.form.get('location'),
                latitude=float(request.form.get('latitude')) if request.form.get('latitude') else None,
                longitude=float(request.form.get('longitude')) if request.form.get('longitude') else None,
                owner_id=current_user.id
            )
            flash('Kiosk creado exitosamente', 'success')
            return redirect(url_for('kiosk.view_kiosk', kiosk_id=kiosk.id))
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            logger.error(f'Error creando kiosk: {str(e)}')
            flash('Error al crear el kiosk', 'error')
    
    return render_template('kiosk/create.html', uuid=str(uuid.uuid4()))

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
    """Vista del historial de ubicaciones de un kiosk."""
    kiosk = kiosk_service.get_kiosk_by_id(kiosk_id)
    if not kiosk:
        flash('Kiosk no encontrado', 'error')
        return redirect(url_for('kiosk.list_kiosks'))
    
    return render_template('kiosk/location_history.html', kiosk=kiosk)

@kiosk_bp.route('/<int:kiosk_id>/update', methods=['GET', 'POST'])
@login_required
@permission_required(UserPermission.UPDATE_KIOSK.value)
def update_kiosk(kiosk_id):
    """Actualizar un kiosk existente."""
    kiosk = kiosk_service.get_kiosk_by_id(kiosk_id)
    if not kiosk:
        flash('Kiosk no encontrado', 'error')
        return redirect(url_for('kiosk.list_kiosks'))
    
    if request.method == 'POST':
        try:
            data = {
                'name': request.form.get('name'),
                'store_name': request.form.get('store_name'),
                'location': request.form.get('location'),
                'latitude': float(request.form.get('latitude')) if request.form.get('latitude') else None,
                'longitude': float(request.form.get('longitude')) if request.form.get('longitude') else None
            }
            
            kiosk_service.update_kiosk(kiosk.id, data)
            flash('Kiosk actualizado exitosamente', 'success')
            return redirect(url_for('kiosk.view_kiosk', kiosk_id=kiosk.id))
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            logger.error(f'Error actualizando kiosk {kiosk_id}: {str(e)}')
            flash('Error al actualizar el kiosk', 'error')
    
    return render_template('kiosk/update.html', kiosk=kiosk)

@kiosk_bp.route('/<uuid:kiosk_uuid>/delete', methods=['POST'])
@login_required
@permission_required(UserPermission.DELETE_KIOSK.value)
def delete_kiosk(kiosk_uuid):
    """Eliminar un kiosk existente."""
    try:
        kiosk = kiosk_service.get_kiosk_by_uuid(kiosk_uuid)
        if not kiosk:
            flash('Kiosk no encontrado', 'error')
            return redirect(url_for('kiosk.list_kiosks'))
        
        kiosk_service.delete_kiosk(kiosk.uuid)
        flash('Kiosk eliminado exitosamente', 'success')
    except Exception as e:
        logger.error(f'Error eliminando kiosk {kiosk_uuid}: {str(e)}')
        flash('Error al eliminar el kiosk', 'error')
    
    return redirect(url_for('kiosk.list_kiosks')) 