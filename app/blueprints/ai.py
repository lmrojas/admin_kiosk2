# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

from flask import Blueprint, render_template, jsonify, request, current_app
from flask_login import login_required
from datetime import datetime

ai_bp = Blueprint('ai', __name__, url_prefix='/ai')

@ai_bp.route('/dashboard')
@login_required
def dashboard():
    """Vista del dashboard de IA"""
    try:
        # Obtener versiones del modelo usando el servicio del contexto
        model_versions = current_app.metrics_service.get_model_versions()
        
        # Si hay versiones disponibles, obtener métricas de la última versión
        if model_versions:
            latest_version = model_versions[0]
            initial_data = current_app.metrics_service.get_initial_metrics(latest_version)
            
            return render_template('ai/dashboard.html',
                                model_versions=model_versions,
                                metrics=initial_data.get('metrics', {}),
                                error_cases=initial_data.get('error_cases', []),
                                class_metrics=initial_data.get('metrics', {}).get('class_metrics', {}),
                                drift={
                                    'kl_divergence': 0.0,
                                    'alerts': []
                                })
        
        # Si no hay versiones, mostrar dashboard vacío con valores por defecto
        return render_template('ai/dashboard.html',
                            model_versions=[],
                            metrics={
                                'accuracy': 0.0,
                                'roc_auc': 0.0,
                                'pr_auc': 0.0,
                                'mean_confidence': 0.0,
                                'confusion_matrix': [],
                                'class_metrics': {}
                            },
                            error_cases=[],
                            class_metrics={},
                            drift={
                                'kl_divergence': 0.0,
                                'alerts': []
                            })
            
    except Exception as e:
        current_app.logger.error(f"Error al cargar el dashboard de IA: {str(e)}")
        # En caso de error, mostrar mensaje pero con valores por defecto
        return render_template('ai/dashboard.html', 
                           error="Error al cargar los datos del dashboard",
                           model_versions=[],
                           metrics={
                               'accuracy': 0.0,
                               'roc_auc': 0.0,
                               'pr_auc': 0.0,
                               'mean_confidence': 0.0,
                               'confusion_matrix': [],
                               'class_metrics': {}
                           },
                           error_cases=[],
                           class_metrics={},
                           drift={
                               'kl_divergence': 0.0,
                               'alerts': []
                           })

@ai_bp.route('/api/metrics')
@login_required
def get_metrics():
    """Obtener métricas del modelo de IA"""
    try:
        # Obtener parámetros de la petición
        version = request.args.get('model_version')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if version and start_date and end_date:
            # Convertir fechas a datetime
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                return jsonify({'error': 'Formato de fecha inválido. Use YYYY-MM-DD'}), 400
                
            # Si se proporcionan todos los parámetros, obtener métricas del período
            metrics_data = current_app.metrics_service.get_metrics_for_period(
                version=version,
                start_date=start_date,
                end_date=end_date
            )
        else:
            # Si no hay parámetros, obtener última versión y sus métricas
            model_versions = current_app.metrics_service.get_model_versions()
            if not model_versions:
                return jsonify({'error': 'No hay versiones del modelo disponibles'})
                
            latest_version = model_versions[0]
            metrics_data = current_app.metrics_service.get_initial_metrics(latest_version)
        
        return jsonify(metrics_data)
        
    except Exception as e:
        current_app.logger.error(f"Error al obtener métricas: {str(e)}")
        return jsonify({'error': 'Error al obtener métricas del modelo'}), 500

@ai_bp.route('/api/predictions/<int:kiosk_id>')
@login_required
def get_predictions(kiosk_id):
    """Obtener predicciones para un kiosk específico"""
    predictions = current_app.ai_service.get_predictions(kiosk_id)
    return jsonify(predictions)

@ai_bp.route('/api/anomalies')
@login_required
def get_anomalies():
    """Obtener anomalías detectadas"""
    anomalies = current_app.ai_service.get_recent_anomalies()
    return jsonify(anomalies)

@ai_bp.route('/api/model/status')
@login_required
def get_model_status():
    """Obtener estado del modelo de IA"""
    status = current_app.metrics_service.get_model_status()
    return jsonify(status) 