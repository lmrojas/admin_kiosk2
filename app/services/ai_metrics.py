"""
Servicio de métricas de IA para Admin Kiosk.
Este código solo puede ser modificado siguiendo lo establecido en 'cura.md' y 'project_custom_structure.txt'
"""

from app.models.ai import ModelMetrics, PredictionLog
from datetime import datetime, timedelta
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve
)
import numpy as np
import logging
from app import db
from sqlalchemy import func, and_
from flask import current_app

logger = logging.getLogger(__name__)

class AIMetricsService:
    """Servicio para gestionar métricas del modelo de IA."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIMetricsService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
    
    def get_model_versions(self):
        """Obtiene las versiones del modelo ordenadas por fecha"""
        try:
            # Subconsulta para obtener el último timestamp por versión
            subquery = db.session.query(
                ModelMetrics.version,
                func.max(ModelMetrics.timestamp).label('max_timestamp')
            ).group_by(ModelMetrics.version).subquery()

            # Consulta principal que incluye el timestamp para poder ordenar por él
            versions = db.session.query(
                ModelMetrics.version,
                ModelMetrics.timestamp
            ).join(
                subquery,
                and_(
                    ModelMetrics.version == subquery.c.version,
                    ModelMetrics.timestamp == subquery.c.max_timestamp
                )
            ).order_by(ModelMetrics.timestamp.desc()).all()

            return [version[0] for version in versions]  # Retornamos solo las versiones
        except Exception as e:
            current_app.logger.error(f"Error al obtener versiones del modelo: {str(e)}")
            return []

    def get_initial_metrics(self, version):
        """Obtiene las métricas iniciales para mostrar."""
        try:
            # Obtener última métrica del modelo usando ORM
            model_metrics = ModelMetrics.query\
                .filter_by(version=version)\
                .order_by(ModelMetrics.timestamp.desc())\
                .first()
            
            if not model_metrics:
                return {
                    'metrics': {
                        'accuracy': 0.0,
                        'roc_auc': 0.0,
                        'pr_auc': 0.0,
                        'mean_confidence': 0.0,
                        'confusion_matrix': [],
                        'class_metrics': {},
                        'roc_curve': {'fpr': [], 'tpr': []},
                        'pr_curve': {'precision': [], 'recall': []}
                    },
                    'error_cases': []
                }
            
            # Obtener predicciones recientes
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            predictions = PredictionLog.query\
                .filter(
                    PredictionLog.model_version == version,
                    PredictionLog.timestamp.between(start_date, end_date)
                ).all()
            
            # Preparar datos para métricas
            y_true = [float(p.actual_value) for p in predictions if p.actual_value is not None]
            y_pred = [float(p.predicted_value) for p in predictions if p.actual_value is not None]
            y_prob = [float(p.confidence) for p in predictions if p.actual_value is not None]
            
            # Calcular métricas
            metrics = self.calculate_model_metrics(y_true, y_pred, y_prob) if y_true else {
                'accuracy': float(model_metrics.metrics.get('accuracy', 0.0)) if model_metrics.metrics else 0.0,
                'roc_auc': float(model_metrics.roc_auc or 0.0),
                'pr_auc': float(model_metrics.pr_auc or 0.0),
                'mean_confidence': float(model_metrics.metrics.get('mean_confidence', 0.0)) if model_metrics.metrics else 0.0,
                'confusion_matrix': model_metrics.confusion_matrix or [],
                'class_metrics': model_metrics.class_metrics or {},
                'roc_curve': {'fpr': [], 'tpr': []},
                'pr_curve': {'precision': [], 'recall': []}
            }
            
            return {
                'metrics': metrics,
                'error_cases': self._get_error_cases(predictions)
            }
        except Exception as e:
            logger.error(f"Error al obtener métricas iniciales: {str(e)}")
            return {
                'metrics': {
                    'accuracy': 0.0,
                    'roc_auc': 0.0,
                    'pr_auc': 0.0,
                    'mean_confidence': 0.0,
                    'confusion_matrix': [],
                    'class_metrics': {},
                    'roc_curve': {'fpr': [], 'tpr': []},
                    'pr_curve': {'precision': [], 'recall': []}
                },
                'error_cases': []
            }

    def get_metrics_for_period(self, version, start_date, end_date):
        """Obtiene métricas para un período específico."""
        try:
            # Obtener predicciones del período usando ORM
            predictions = PredictionLog.query\
                .filter(
                    PredictionLog.model_version == version,
                    PredictionLog.timestamp.between(start_date, end_date)
                ).all()
            
            if not predictions:
                return {'error': 'No hay datos suficientes para el período seleccionado'}
            
            # Preparar datos para métricas
            y_true = [float(p.actual_value) for p in predictions if p.actual_value is not None]
            y_pred = [float(p.predicted_value) for p in predictions if p.actual_value is not None]
            y_prob = [float(p.confidence) for p in predictions if p.actual_value is not None]
            
            if not y_true:
                return {'error': 'No hay datos suficientes para el período seleccionado'}
            
            # Calcular métricas
            metrics = self.calculate_model_metrics(y_true, y_pred, y_prob)
            
            # Obtener casos con error
            error_cases = self._get_error_cases(predictions)
            
            return {
                'metrics': metrics,
                'error_cases': error_cases
            }
            
        except Exception as e:
            logger.error(f"Error al obtener métricas del período: {str(e)}")
            return {'error': 'Error al calcular métricas'}

    def calculate_model_metrics(self, y_true, y_pred, y_prob):
        """Calcula métricas del modelo."""
        try:
            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = float(auc(fpr, tpr))
            
            # Calcular curva PR
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall, precision))
            
            metrics = {
                'accuracy': float(np.mean(np.array(y_true) == np.array(y_pred))),
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'mean_confidence': float(np.mean(y_prob)),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                },
                'pr_curve': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }
            }
            
            # Métricas por clase
            metrics['class_metrics'] = {
                'precision': [float(precision_score(y_true, y_pred, pos_label=i)) for i in [0, 1]],
                'recall': [float(recall_score(y_true, y_pred, pos_label=i)) for i in [0, 1]],
                'f1': [float(f1_score(y_true, y_pred, pos_label=i)) for i in [0, 1]]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error al calcular métricas del modelo: {str(e)}")
            return {
                'accuracy': 0.0,
                'roc_auc': 0.0,
                'pr_auc': 0.0,
                'mean_confidence': 0.0,
                'confusion_matrix': [],
                'class_metrics': {},
                'roc_curve': {'fpr': [], 'tpr': []},
                'pr_curve': {'precision': [], 'recall': []}
            }

    def _get_error_cases(self, predictions, limit=10):
        """Obtiene los casos con error más recientes."""
        error_cases = []
        for pred in predictions:
            if pred.actual_value is not None and pred.predicted_value != pred.actual_value:
                error_cases.append({
                    'timestamp': pred.timestamp.isoformat(),
                    'predicted': int(pred.predicted_value),
                    'actual': int(pred.actual_value),
                    'confidence': float(pred.confidence),
                    'features': pred.features
                })
                if len(error_cases) >= limit:
                    break
        return error_cases 