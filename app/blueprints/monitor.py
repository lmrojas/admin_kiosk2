# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

from flask import Blueprint, jsonify, render_template
from flask_login import login_required
from app.services.monitoring_service import MonitoringService
from app.services.security_service import SecurityService

monitor_bp = Blueprint('monitor', __name__, url_prefix='/monitor')
monitoring_service = MonitoringService()
security_service = SecurityService()

@monitor_bp.route('/')
@login_required
@security_service.rate_limit_decorator()
def dashboard():
    """Renderiza el dashboard principal de monitoreo"""
    return render_template('monitor/dashboard.html')

@monitor_bp.route('/metrics')
@login_required
@security_service.rate_limit_decorator()
def get_metrics():
    """Obtiene métricas actuales del sistema"""
    metrics = monitoring_service.get_system_metrics()
    monitoring_service.cache_metrics(metrics)
    return jsonify(metrics)

@monitor_bp.route('/kiosk/<int:kiosk_id>/health')
@login_required
@security_service.rate_limit_decorator()
def get_kiosk_health(kiosk_id):
    """Obtiene métricas de salud de un kiosk específico"""
    health = monitoring_service.get_kiosk_health(kiosk_id)
    return jsonify(health)

@monitor_bp.route('/alerts')
@login_required
@security_service.rate_limit_decorator()
def get_alerts():
    """Obtiene alertas recientes del sistema"""
    hours = request.args.get('hours', 24, type=int)
    alerts = monitoring_service.get_alerts(hours)
    return jsonify(alerts)

@monitor_bp.route('/metrics/history')
@login_required
@security_service.rate_limit_decorator()
def get_metrics_history():
    """Obtiene historial de métricas cacheadas"""
    minutes = request.args.get('minutes', 60, type=int)
    metrics = monitoring_service.get_cached_metrics(minutes)
    return jsonify(metrics) 