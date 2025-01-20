# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required
from app.services.kiosk_ai_service import KioskAIService
from app.services.ai_metrics import AIMetricsService

ai_bp = Blueprint('ai', __name__, url_prefix='/ai')
ai_service = KioskAIService()
metrics_service = AIMetricsService()

@ai_bp.route('/dashboard')
@login_required
def dashboard():
    """Vista del dashboard de IA"""
    return render_template('ai/dashboard.html')

@ai_bp.route('/metrics')
@login_required
def get_metrics():
    """Obtener métricas del modelo de IA"""
    metrics = metrics_service.get_current_metrics()
    return jsonify(metrics)

@ai_bp.route('/predictions/<int:kiosk_id>')
@login_required
def get_predictions(kiosk_id):
    """Obtener predicciones para un kiosk específico"""
    predictions = ai_service.get_predictions(kiosk_id)
    return jsonify(predictions)

@ai_bp.route('/anomalies')
@login_required
def get_anomalies():
    """Obtener anomalías detectadas"""
    anomalies = ai_service.get_recent_anomalies()
    return jsonify(anomalies)

@ai_bp.route('/model/status')
@login_required
def get_model_status():
    """Obtener estado del modelo de IA"""
    status = metrics_service.get_model_status()
    return jsonify(status) 