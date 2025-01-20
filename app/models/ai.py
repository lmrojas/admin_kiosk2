"""
Modelos para el sistema de IA de Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from datetime import datetime
from app import db
from sqlalchemy.dialects.postgresql import JSON

class ModelMetrics(db.Model):
    """Modelo para almacenar métricas del modelo de IA."""
    __tablename__ = 'model_metrics'

    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50), nullable=False, 
                     comment='Identificador de la versión del modelo')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow,
                       comment='Fecha y hora del registro de métricas')
    metrics = db.Column(JSON, comment='Métricas detalladas del modelo')
    roc_auc = db.Column(db.Float, comment='Área bajo la curva ROC')
    pr_auc = db.Column(db.Float, comment='Área bajo la curva Precisión-Recall')
    confusion_matrix = db.Column(JSON, comment='Matriz de confusión del modelo')
    class_metrics = db.Column(JSON, comment='Métricas detalladas por clase')
    calibration_metrics = db.Column(JSON, comment='Métricas de calibración')

class PredictionLog(db.Model):
    """Modelo para registrar predicciones y monitorear drift."""
    __tablename__ = 'prediction_logs'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    model_version = db.Column(db.String(50), nullable=False)
    features = db.Column(JSON)
    predicted_value = db.Column(db.Float)
    actual_value = db.Column(db.Float)
    confidence = db.Column(db.Float)
    prediction_time = db.Column(db.Float)
    extra_data = db.Column(JSON)

class DriftMetrics(db.Model):
    """Modelo para almacenar métricas de drift."""
    __tablename__ = 'drift_metrics'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    model_version = db.Column(db.String(50), nullable=False)
    analysis_window = db.Column(db.Integer)
    distribution_shift = db.Column(db.Float)
    feature_drift = db.Column(JSON)
    performance_metrics = db.Column(JSON)
    alerts = db.Column(JSON)
    severity = db.Column(db.String(20)) 