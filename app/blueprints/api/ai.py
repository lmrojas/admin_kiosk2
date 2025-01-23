"""
API endpoints para gestión del modelo de IA.
Sigue el patrón MVT + S.
"""

from flask import Blueprint, jsonify, request, current_app
from app.services.kiosk_ai_service import KioskAIService
from app.services.auth_service import admin_required
from app.models.ai import PredictionLog
from datetime import datetime, timedelta

ai_api = Blueprint('ai_api', __name__)
ai_service = KioskAIService()

@ai_api.route('/metrics', methods=['GET'])
@admin_required
def get_metrics():
    """Obtiene métricas actuales del modelo."""
    try:
        metrics = ai_service.get_model_metrics()
        if metrics:
            return jsonify({
                'success': True,
                'metrics': metrics
            }), 200
        return jsonify({
            'success': False,
            'message': 'Error obteniendo métricas'
        }), 500
    except Exception as e:
        current_app.logger.error(f"Error en get_metrics: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@ai_api.route('/auto-training/toggle', methods=['POST'])
@admin_required
def toggle_auto_training():
    """Activa o desactiva el autoentrenamiento."""
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        if ai_service.toggle_auto_training(enabled):
            return jsonify({
                'success': True,
                'message': f'Autoentrenamiento {"activado" if enabled else "desactivado"}',
                'auto_training_enabled': enabled
            }), 200
            
        return jsonify({
            'success': False,
            'message': 'Error al cambiar estado de autoentrenamiento'
        }), 500
        
    except Exception as e:
        current_app.logger.error(f"Error en toggle_auto_training: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@ai_api.route('/auto-training/params', methods=['POST'])
@admin_required
def set_auto_training_params():
    """Configura parámetros de autoentrenamiento."""
    try:
        data = request.get_json()
        min_samples = data.get('min_samples')
        check_interval = data.get('check_interval')
        
        if ai_service.set_auto_training_params(min_samples, check_interval):
            return jsonify({
                'success': True,
                'message': 'Parámetros actualizados',
                'params': {
                    'min_samples': ai_service.min_samples_for_training,
                    'check_interval': ai_service.training_check_interval
                }
            }), 200
            
        return jsonify({
            'success': False,
            'message': 'Error al actualizar parámetros'
        }), 500
        
    except Exception as e:
        current_app.logger.error(f"Error en set_auto_training_params: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@ai_api.route('/predictions/history', methods=['GET'])
@admin_required
def get_prediction_history():
    """Obtiene historial de predicciones con filtros."""
    try:
        # Parámetros de filtrado
        days = request.args.get('days', 7, type=int)
        limit = min(request.args.get('limit', 100, type=int), 1000)
        only_anomalies = request.args.get('only_anomalies', False, type=bool)
        
        # Calcular fecha desde
        date_from = datetime.now() - timedelta(days=days)
        
        # Construir query base
        query = PredictionLog.query.filter(PredictionLog.timestamp >= date_from)
        
        # Filtrar solo anomalías si se solicita
        if only_anomalies:
            query = query.filter(PredictionLog.predicted_value == True)
            
        # Ordenar y limitar
        predictions = query.order_by(PredictionLog.timestamp.desc()).limit(limit).all()
        
        return jsonify({
            'success': True,
            'predictions': [p.to_dict() for p in predictions]
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error en get_prediction_history: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500 