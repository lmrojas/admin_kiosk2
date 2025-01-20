# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y SOLAMENTE
# SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from flask import Blueprint, jsonify, request, render_template
from app.services.dashboard_service import DashboardService
from app.decorators import jwt_required, permission_required

dashboard_bp = Blueprint('dashboard', __name__)
dashboard_service = DashboardService()

# Rutas de Vista
@dashboard_bp.route('/dashboard', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def index():
    """Vista principal del dashboard"""
    return render_template('dashboard/index.html')

@dashboard_bp.route('/dashboard/logs', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def logs():
    """Vista de logs"""
    return render_template('dashboard/logs.html')

@dashboard_bp.route('/dashboard/security', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def security():
    """Vista de seguridad"""
    return render_template('dashboard/security.html')

@dashboard_bp.route('/dashboard/backups', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def backups():
    """Vista de backups"""
    return render_template('dashboard/backups.html')

# Rutas de API
@dashboard_bp.route('/api/dashboard/status', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def get_system_status():
    """Endpoint para obtener el estado del sistema"""
    return jsonify(dashboard_service.get_system_status())

@dashboard_bp.route('/api/dashboard/metrics', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def get_system_metrics():
    """Endpoint para obtener métricas del sistema"""
    return jsonify(dashboard_service.get_system_metrics())

@dashboard_bp.route('/api/dashboard/alerts', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def get_recent_alerts():
    """Endpoint para obtener alertas recientes"""
    hours = request.args.get('hours', default=24, type=int)
    return jsonify(dashboard_service.get_recent_alerts(hours=hours))

@dashboard_bp.route('/api/dashboard/backups', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def get_backup_status():
    """Endpoint para obtener estado de backups"""
    return jsonify(dashboard_service.get_backup_status())

@dashboard_bp.route('/api/dashboard/logs/summary', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def get_log_summary():
    """Endpoint para obtener resumen de logs"""
    days = request.args.get('days', default=7, type=int)
    return jsonify(dashboard_service.get_log_summary(days=days))

@dashboard_bp.route('/api/dashboard/security/events', methods=['GET'])
@jwt_required
@permission_required('view_dashboard')
def get_security_events():
    """Endpoint para obtener eventos de seguridad"""
    days = request.args.get('days', default=7, type=int)
    return jsonify(dashboard_service.get_security_events(days=days))

@dashboard_bp.errorhandler(Exception)
def handle_error(error):
    """Manejador de errores para las rutas del dashboard"""
    return jsonify({
        'error': str(error),
        'type': error.__class__.__name__
    }), 500 