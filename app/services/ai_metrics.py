"""
Servicio de métricas de IA para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from app.models.ai import ModelMetrics, PredictionLog, DriftMetrics
from datetime import datetime, timedelta
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score
)
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AIMetricsService:
    """Servicio para gestionar métricas del modelo de IA."""

    def get_model_versions(self):
        """Obtiene las versiones disponibles del modelo."""
        return list(ModelMetrics.objects.values_list(
            'version', flat=True
        ).distinct().order_by('-timestamp'))

    def get_initial_metrics(self, version):
        """Obtiene las métricas iniciales para mostrar."""
        try:
            # Obtener última métrica del modelo
            model_metrics = ModelMetrics.objects.filter(
                version=version
            ).latest('timestamp')
            
            # Obtener predicciones recientes
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            predictions = PredictionLog.objects.filter(
                model_version=version,
                timestamp__range=(start_date, end_date)
            )
            
            # Obtener últimas métricas de drift
            drift_metrics = DriftMetrics.objects.filter(
                model_version=version
            ).latest('timestamp')
            
            return {
                'metrics': model_metrics.metrics,
                'drift': drift_metrics.to_dict() if drift_metrics else {},
                'error_cases': self._get_error_cases(predictions)
            }
        except Exception as e:
            logger.error(f"Error al obtener métricas iniciales: {str(e)}")
            return {'error': 'Error al obtener métricas iniciales'}

    def get_metrics_for_period(self, version, start_date, end_date):
        """Obtiene métricas para un período específico."""
        try:
            # Obtener predicciones del período
            predictions = PredictionLog.objects.filter(
                model_version=version,
                timestamp__range=(start_date, end_date)
            )
            
            # Preparar datos para métricas
            y_true = [p.actual_value for p in predictions if p.actual_value is not None]
            y_pred = [p.predicted_value for p in predictions if p.actual_value is not None]
            y_prob = [p.confidence for p in predictions if p.actual_value is not None]
            
            if not y_true:
                return {'error': 'No hay datos suficientes para el período seleccionado'}
            
            # Calcular métricas
            metrics = self.calculate_model_metrics(y_true, y_pred, y_prob)
            
            # Analizar drift
            drift = self.analyze_prediction_drift(
                version=version,
                window_days=(end_date - start_date).days
            )
            
            # Obtener casos con error
            error_cases = self._get_error_cases(predictions)
            
            return {
                'metrics': metrics,
                'drift': drift,
                'error_cases': error_cases
            }
            
        except Exception as e:
            logger.error(f"Error al obtener métricas del período: {str(e)}")
            return {'error': 'Error al calcular métricas'}

    def calculate_model_metrics(self, y_true, y_pred, y_prob):
        """Calcula métricas del modelo."""
        try:
            metrics = {
                'accuracy': np.mean(np.array(y_true) == np.array(y_pred)),
                'roc_auc': roc_auc_score(y_true, y_prob),
                'mean_confidence': np.mean(y_prob),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
            # Calcular curva PR
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = auc(recall, precision)
            
            # Métricas por clase
            metrics['class_metrics'] = {
                'precision': precision_score(y_true, y_pred, average=None).tolist(),
                'recall': recall_score(y_true, y_pred, average=None).tolist(),
                'f1': f1_score(y_true, y_pred, average=None).tolist()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error al calcular métricas del modelo: {str(e)}")
            return {'error': 'Error al calcular métricas del modelo'}

    def analyze_prediction_drift(self, version, window_days):
        """Analiza el drift en las predicciones."""
        try:
            # Obtener métricas de drift existentes
            drift_metrics = DriftMetrics.objects.filter(
                model_version=version,
                analysis_window=window_days
            ).latest('timestamp')
            
            if drift_metrics:
                return drift_metrics.to_dict()
            
            # Si no hay métricas existentes, calcular nuevas
            return self._calculate_new_drift_metrics(version, window_days)
            
        except Exception as e:
            logger.error(f"Error al analizar drift: {str(e)}")
            return {'error': 'Error al analizar drift'}

    def _calculate_new_drift_metrics(self, version, window_days):
        """Calcula nuevas métricas de drift."""
        # Implementar cálculo de drift según necesidades específicas
        return {
            'distribution_shift': self._calculate_distribution_shift(version),
            'feature_drift': self._calculate_feature_drift(version),
            'performance_decay': self._calculate_performance_decay(version, window_days)
        }

    def _get_error_cases(self, predictions, limit=10):
        """Obtiene los casos con mayor error."""
        error_cases = []
        
        for pred in predictions:
            if pred.actual_value is not None:
                error = abs(pred.actual_value - pred.predicted_value)
                error_cases.append({
                    'id': pred.id,
                    'predicted': pred.predicted_value,
                    'actual': pred.actual_value,
                    'confidence': pred.confidence,
                    'error_margin': error
                })
        
        # Ordenar por error y tomar los top N
        error_cases.sort(key=lambda x: x['error_margin'], reverse=True)
        return error_cases[:limit]

    def _calculate_distribution_shift(self, version):
        """Calcula el cambio en la distribución de predicciones."""
        try:
            # Obtener predicciones del último período
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            predictions = PredictionLog.objects.filter(
                model_version=version,
                timestamp__range=(start_date, end_date)
            ).order_by('timestamp')
            
            if not predictions:
                return {}
                
            # Dividir en dos períodos
            mid_point = len(predictions) // 2
            period1 = predictions[:mid_point]
            period2 = predictions[mid_point:]
            
            # Calcular distribuciones
            dist1 = self._get_prediction_distribution(period1)
            dist2 = self._get_prediction_distribution(period2)
            
            # Calcular KL divergence
            kl_div = self._calculate_kl_divergence(dist1, dist2)
            
            return {
                'kl_divergence': float(kl_div),
                'distribution_period1': dist1,
                'distribution_period2': dist2
            }
            
        except Exception as e:
            logger.error(f"Error al calcular distribution shift: {str(e)}")
            return {}
            
    def _calculate_feature_drift(self, version):
        """Calcula el drift en las características."""
        try:
            # Obtener predicciones con features
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            predictions = PredictionLog.objects.filter(
                model_version=version,
                timestamp__range=(start_date, end_date)
            ).order_by('timestamp')
            
            if not predictions:
                return {}
                
            # Extraer features y dividir en períodos
            mid_point = len(predictions) // 2
            period1 = predictions[:mid_point]
            period2 = predictions[mid_point:]
            
            # Calcular estadísticas por feature
            feature_stats = {}
            if period1 and period2:
                features = period1[0].features.keys()
                for feature in features:
                    values1 = [p.features[feature] for p in period1]
                    values2 = [p.features[feature] for p in period2]
                    
                    mean1, std1 = np.mean(values1), np.std(values1)
                    mean2, std2 = np.mean(values2), np.std(values2)
                    
                    # Calcular score de drift
                    mean_change = abs(mean2 - mean1) / (abs(mean1) + 1e-10)
                    std_change = abs(std2 - std1) / (abs(std1) + 1e-10)
                    drift_score = mean_change + std_change
                    
                    feature_stats[feature] = {
                        'drift_score': float(drift_score),
                        'mean_change': float(mean_change),
                        'std_change': float(std_change)
                    }
            
            return feature_stats
            
        except Exception as e:
            logger.error(f"Error al calcular feature drift: {str(e)}")
            return {}

    def _calculate_performance_decay(self, version, window_days):
        """Calcula la degradación del rendimiento."""
        try:
            # Obtener predicciones del período
            end_date = datetime.now()
            start_date = end_date - timedelta(days=window_days)
            
            predictions = PredictionLog.objects.filter(
                model_version=version,
                timestamp__range=(start_date, end_date)
            ).order_by('timestamp')
            
            if not predictions:
                return {}
                
            # Dividir en ventanas temporales
            window_size = max(len(predictions) // 4, 1)
            windows = [predictions[i:i+window_size] 
                      for i in range(0, len(predictions), window_size)]
            
            # Calcular métricas por ventana
            window_metrics = []
            for i, window in enumerate(windows):
                if not window:
                    continue
                    
                y_true = [p.actual_value for p in window if p.actual_value is not None]
                y_pred = [p.predicted_value for p in window if p.actual_value is not None]
                
                if not y_true:
                    continue
                
                accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                confidence = np.mean([p.confidence for p in window])
                
                window_metrics.append({
                    'window': i,
                    'start_date': window[0].timestamp.isoformat(),
                    'end_date': window[-1].timestamp.isoformat(),
                    'accuracy': float(accuracy),
                    'mean_confidence': float(confidence),
                    'sample_size': len(window)
                })
            
            # Calcular tendencia
            if len(window_metrics) >= 2:
                accuracies = [w['accuracy'] for w in window_metrics]
                slope = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            else:
                slope = 0.0
            
            return {
                'window_metrics': window_metrics,
                'decay_rate': float(slope)
            }
            
        except Exception as e:
            logger.error(f"Error al calcular performance decay: {str(e)}")
            return {}
            
    def _get_prediction_distribution(self, predictions):
        """Calcula la distribución de predicciones."""
        if not predictions:
            return {}
            
        values = [p.predicted_value for p in predictions]
        unique, counts = np.unique(values, return_counts=True)
        return dict(zip(unique.tolist(), (counts/len(values)).tolist()))
        
    def _calculate_kl_divergence(self, dist1, dist2):
        """Calcula la divergencia KL entre dos distribuciones."""
        if not dist1 or not dist2:
            return 0.0
            
        # Asegurar que ambas distribuciones tengan las mismas clases
        all_classes = set(dist1.keys()) | set(dist2.keys())
        
        # Añadir suavizado para evitar divisiones por cero
        epsilon = 1e-10
        
        kl_div = 0
        for c in all_classes:
            p = dist1.get(c, epsilon)
            q = dist2.get(c, epsilon)
            kl_div += p * np.log(p/q)
        
        return kl_div 