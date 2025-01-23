# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union
import logging
from datetime import datetime, timedelta
import time
from flask import current_app
from app import socketio, db
from app.models.ai import PredictionLog
import os
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve
)

class SimpleAnomalyDetector(nn.Module):
    """
    Modelo de red neuronal simple para detección de anomalías en kiosks.
    """
    def __init__(self, input_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class KioskAIService:
    """Servicio para gestionar el modelo de IA de los kiosks."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KioskAIService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.model = None
            self._initialized = True
            self.is_training = False
            self.training_progress = 0
            self.current_epoch = 0
            self.total_epochs = 50
            self.batch_size = 32
            self.learning_rate = 0.001
            self.training_loss = 0.0
            self.model_version = 'v1.0.0'
            self.anomaly_threshold = 0.7
            self.auto_training_enabled = True
            self.min_samples_for_training = 100
            self.last_training_check = datetime.now()
            self.training_check_interval = 3600  # 1 hora
            self._load_model()
            
    def _load_model(self):
        """Carga o inicializa el modelo."""
        try:
            # TODO: Implementar carga desde archivo
            self.model = SimpleAnomalyDetector()
            logging.info("Modelo de IA inicializado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar modelo: {str(e)}")
            self.model = SimpleAnomalyDetector()  # Fallback a modelo nuevo
    
    def _save_prediction(self, prediction: Dict) -> None:
        """Guarda una predicción en la base de datos y verifica autoentrenamiento."""
        try:
            with current_app.app_context():
                pred_log = PredictionLog(
                    timestamp=prediction['timestamp'],
                    model_version=prediction.get('model_version', 'v1.0.0'),
                    features=prediction['features'],
                    predicted_value=prediction['predicted_value'],
                    actual_value=prediction.get('actual_value'),
                    confidence=prediction['confidence'],
                    prediction_time=prediction['prediction_time']
                )
                db.session.add(pred_log)
                db.session.commit()

                # Verificar autoentrenamiento después de guardar
                self.check_auto_training()

        except Exception as e:
            logging.error(f"Error al guardar predicción: {str(e)}")
            if 'db' in locals():
                db.session.rollback()

    def predict_anomaly(self, kiosk_data: Dict[str, Union[float, int]]) -> Dict:
        """Predice anomalías basado en datos recibidos de una kiosk."""
        try:
            # Validar y normalizar datos
            features = [
                kiosk_data.get('cpu_usage', 0.0) / 100.0,
                kiosk_data.get('memory_usage', 0.0) / 100.0,
                min(kiosk_data.get('network_latency', 0.0) / 300.0, 1.0)
            ]
            
            # Agregar diferencia de ubicación si está disponible
            location_diff = kiosk_data.get('location_difference', 0.0)
            if location_diff is not None:
                # Normalizar diferencia de ubicación (asumiendo máximo 1000m como umbral)
                features.append(min(location_diff / 1000.0, 1.0))
            
            # Convertir a tensor y predecir
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            start_time = time.time()
            
            with torch.no_grad():
                anomaly_prob = self.model(input_tensor).item()
            
            # Calcular tiempo de predicción
            prediction_time = time.time() - start_time
            
            # Determinar si es anomalía
            is_anomaly = anomaly_prob > 0.7
            
            # Crear registro de predicción
            prediction = {
                'timestamp': datetime.now(),
                'kiosk_id': kiosk_data.get('kiosk_id'),
                'features': {
                    'cpu_usage': kiosk_data.get('cpu_usage'),
                    'memory_usage': kiosk_data.get('memory_usage'),
                    'network_latency': kiosk_data.get('network_latency'),
                    'location_difference': location_diff
                },
                'predicted_value': int(is_anomaly),
                'confidence': anomaly_prob,
                'prediction_time': prediction_time
            }
            
            # Guardar predicción
            self._save_prediction(prediction)
            
            # Emitir resultado por WebSocket
            socketio.emit('kiosk_prediction', {
                'kiosk_id': kiosk_data.get('kiosk_id'),
                'timestamp': prediction['timestamp'].isoformat(),
                'metrics': prediction['features'],
                'is_anomaly': bool(prediction['predicted_value']),
                'confidence': prediction['confidence']
            })
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error en predicción: {str(e)}")
            return None

    def detect_location_anomaly(self, kiosk_id: int) -> Dict:
        """
        Detecta anomalías específicas de ubicación para un kiosk.
        """
        try:
            from app.models.kiosk import Kiosk
            kiosk = Kiosk.query.get(kiosk_id)
            if not kiosk:
                return None
                
            distance, time_diff = kiosk.get_location_difference()
            if distance is None:
                return None
                
            # Preparar datos para predicción
            kiosk_data = {
                'kiosk_id': kiosk_id,
                'location_difference': distance,
                'cpu_usage': 0.0,  # Valores neutrales para otros features
                'memory_usage': 0.0,
                'network_latency': 0.0
            }
            
            # Usar el modelo existente para predecir
            prediction = self.predict_anomaly(kiosk_data)
            
            if prediction and prediction['predicted_value']:
                # Emitir alerta específica de ubicación
                socketio.emit('location_anomaly', {
                    'kiosk_id': kiosk_id,
                    'distance': distance,
                    'time_diff': time_diff,
                    'confidence': prediction['confidence']
                })
                
            return prediction
            
        except Exception as e:
            logging.error(f"Error en detección de anomalía de ubicación: {str(e)}")
            return None 

    def get_model_metrics(self) -> Dict:
        """Obtiene las métricas actuales del modelo."""
        try:
            # Calcular métricas desde PredictionLog
            from app.models.ai import PredictionLog
            from sqlalchemy import func
            
            with current_app.app_context():
                # Obtener predicciones recientes (últimos 7 días)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                predictions = PredictionLog.query.filter(
                    PredictionLog.model_version == self.model_version,
                    PredictionLog.timestamp.between(start_date, end_date)
                ).all()
                
                # Preparar datos para métricas
                if predictions:
                    y_true = [float(p.actual_value) for p in predictions if p.actual_value is not None]
                    y_pred = [float(p.predicted_value) for p in predictions if p.actual_value is not None]
                    y_prob = [float(p.confidence) for p in predictions if p.actual_value is not None]
                    
                    if y_true:
                        # Calcular curva ROC
                        fpr, tpr, _ = roc_curve(y_true, y_prob)
                        roc_auc = float(auc(fpr, tpr))
                        
                        # Calcular curva PR
                        precision, recall, _ = precision_recall_curve(y_true, y_prob)
                        pr_auc = float(auc(recall, precision))
                        
                        # Calcular matriz de confusión y métricas por clase
                        conf_matrix = confusion_matrix(y_true, y_pred).tolist()
                        class_metrics = {
                            'precision': [float(precision_score(y_true, y_pred, pos_label=i)) for i in [0, 1]],
                            'recall': [float(recall_score(y_true, y_pred, pos_label=i)) for i in [0, 1]],
                            'f1': [float(f1_score(y_true, y_pred, pos_label=i)) for i in [0, 1]]
                        }
                        
                        # Tiempo promedio de predicción
                        avg_prediction_time = db.session.query(
                            func.avg(PredictionLog.prediction_time)
                        ).scalar() or 0
                        
                        # Calcular tasa de aciertos
                        accuracy_rate = (sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) * 100)
                        
                        return {
                            'model_version': self.model_version,
                            'avg_prediction_time': round(avg_prediction_time * 1000, 2),  # ms
                            'accuracy_rate': round(accuracy_rate, 2),
                            'roc_auc': round(roc_auc, 3),
                            'pr_auc': round(pr_auc, 3),
                            'mean_confidence': round(float(np.mean(y_prob)), 3),
                            'confusion_matrix': conf_matrix,
                            'class_metrics': class_metrics,
                            'anomaly_threshold': self.anomaly_threshold,
                            'is_training': self.is_training,
                            'training_progress': self.training_progress,
                            'training_loss': round(self.training_loss, 4),
                            'current_epoch': self.current_epoch,
                            'total_epochs': self.total_epochs,
                            'batch_size': self.batch_size,
                            'learning_rate': self.learning_rate,
                            'error_cases': self._get_error_cases(predictions)
                        }
                
                # Si no hay predicciones, retornar valores por defecto
                return {
                    'model_version': self.model_version,
                    'avg_prediction_time': 0,
                    'accuracy_rate': 0.0,
                    'roc_auc': 0.0,
                    'pr_auc': 0.0,
                    'mean_confidence': 0.0,
                    'confusion_matrix': [[0, 0], [0, 0]],
                    'class_metrics': {
                        'precision': [0.0, 0.0],
                        'recall': [0.0, 0.0],
                        'f1': [0.0, 0.0]
                    },
                    'anomaly_threshold': self.anomaly_threshold,
                    'is_training': self.is_training,
                    'training_progress': self.training_progress,
                    'training_loss': self.training_loss,
                    'current_epoch': self.current_epoch,
                    'total_epochs': self.total_epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'error_cases': []
                }
                
        except Exception as e:
            logging.error(f"Error obteniendo métricas del modelo: {str(e)}")
            return None

    def _get_error_cases(self, predictions, limit=10):
        """Obtiene los casos con error más recientes."""
        error_cases = []
        try:
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
        except Exception as e:
            logging.error(f"Error obteniendo casos de error: {str(e)}")
        return error_cases

    def start_training(self):
        """Inicia el entrenamiento del modelo."""
        if self.is_training:
            return False
            
        try:
            self.is_training = True
            self.training_progress = 0
            self.current_epoch = 0
            
            # Emitir estado inicial
            self._emit_training_status()
            
            # Iniciar entrenamiento en thread separado
            import threading
            training_thread = threading.Thread(target=self._train_model)
            training_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Error iniciando entrenamiento: {str(e)}")
            self.is_training = False
            return False

    def stop_training(self):
        """Detiene el entrenamiento del modelo."""
        self.is_training = False
        return True

    def _train_model(self):
        """Proceso de entrenamiento del modelo."""
        try:
            # Obtener datos de entrenamiento
            train_data = self._get_training_data()
            if not train_data:
                self.is_training = False
                return
                
            # Configurar optimizador
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.BCELoss()
            
            # Entrenar por épocas
            for epoch in range(self.total_epochs):
                if not self.is_training:
                    break
                    
                self.current_epoch = epoch + 1
                epoch_loss = 0.0
                
                # Entrenar en batches
                for i in range(0, len(train_data['features']), self.batch_size):
                    if not self.is_training:
                        break
                        
                    # Obtener batch
                    batch_features = train_data['features'][i:i + self.batch_size]
                    batch_labels = train_data['labels'][i:i + self.batch_size]
                    
                    # Convertir a tensores
                    features = torch.tensor(batch_features, dtype=torch.float32)
                    labels = torch.tensor(batch_labels, dtype=torch.float32)
                    
                    # Forward pass
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Actualizar métricas
                self.training_loss = epoch_loss / (len(train_data['features']) / self.batch_size)
                self.training_progress = int((self.current_epoch / self.total_epochs) * 100)
                
                # Emitir estado
                self._emit_training_status()
            
            # Actualizar versión del modelo
            if self.is_training:
                self.model_version = f'v1.1.{int(time.time())}'
                self._save_model()
            
            self.is_training = False
            self._emit_training_status()
            
        except Exception as e:
            logging.error(f"Error en entrenamiento: {str(e)}")
            self.is_training = False
            self._emit_training_status()

    def _get_training_data(self) -> Dict:
        """Obtiene datos de entrenamiento desde PredictionLog."""
        try:
            from app.models.ai import PredictionLog
            
            with current_app.app_context():
                logs = PredictionLog.query.all()
                
                features = []
                labels = []
                
                for log in logs:
                    if all(k in log.features for k in ['cpu_usage', 'memory_usage', 'network_latency']):
                        features.append([
                            log.features['cpu_usage'] / 100.0,
                            log.features['memory_usage'] / 100.0,
                            min(log.features['network_latency'] / 300.0, 1.0)
                        ])
                        labels.append([float(log.actual_value)])
                
                return {
                    'features': features,
                    'labels': labels
                }
                
        except Exception as e:
            logging.error(f"Error obteniendo datos de entrenamiento: {str(e)}")
            return None

    def _emit_training_status(self):
        """Emite el estado actual del entrenamiento vía WebSocket."""
        try:
            socketio.emit('training_update', {
                'is_training': self.is_training,
                'progress': self.training_progress,
                'loss': round(self.training_loss, 4),
                'current_epoch': self.current_epoch,
                'total_epochs': self.total_epochs
            })
        except Exception as e:
            logging.error(f"Error emitiendo estado de entrenamiento: {str(e)}")

    def _save_model(self):
        """Guarda el modelo entrenado."""
        try:
            model_path = os.path.join(current_app.config['MODEL_PATH'], f'model_{self.model_version}.pt')
            torch.save(self.model.state_dict(), model_path)
            logging.info(f"Modelo guardado: {model_path}")
        except Exception as e:
            logging.error(f"Error guardando modelo: {str(e)}")

    def check_auto_training(self):
        """Verifica si es necesario realizar autoentrenamiento."""
        try:
            # Solo verificar si el autoentrenamiento está habilitado y no hay entrenamiento en curso
            if not self.auto_training_enabled or self.is_training:
                return False

            # Verificar si ha pasado suficiente tiempo desde la última verificación
            now = datetime.now()
            if (now - self.last_training_check).total_seconds() < self.training_check_interval:
                return False

            self.last_training_check = now

            # Verificar cantidad de nuevos datos desde última versión del modelo
            with current_app.app_context():
                from app.models.ai import PredictionLog
                
                # Contar registros nuevos desde la última versión
                new_records = PredictionLog.query.filter(
                    PredictionLog.model_version == self.model_version
                ).count()

                if new_records >= self.min_samples_for_training:
                    logging.info(f"Iniciando autoentrenamiento con {new_records} nuevas muestras")
                    return self.start_training()

            return False

        except Exception as e:
            logging.error(f"Error verificando autoentrenamiento: {str(e)}")
            return False

    def toggle_auto_training(self, enabled: bool) -> bool:
        """Activa o desactiva el autoentrenamiento."""
        try:
            self.auto_training_enabled = enabled
            logging.info(f"Autoentrenamiento {'activado' if enabled else 'desactivado'}")
            return True
        except Exception as e:
            logging.error(f"Error al cambiar estado de autoentrenamiento: {str(e)}")
            return False

    def set_auto_training_params(self, min_samples: int = None, check_interval: int = None) -> bool:
        """Configura parámetros de autoentrenamiento."""
        try:
            if min_samples is not None:
                self.min_samples_for_training = max(50, min_samples)
            if check_interval is not None:
                self.training_check_interval = max(300, check_interval)  # Mínimo 5 minutos
            logging.info(f"Parámetros de autoentrenamiento actualizados: min_samples={self.min_samples_for_training}, check_interval={self.training_check_interval}")
            return True
        except Exception as e:
            logging.error(f"Error al configurar parámetros de autoentrenamiento: {str(e)}")
            return False 