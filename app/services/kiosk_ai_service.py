# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union
import logging
from datetime import datetime
import time
from flask import current_app
from app import socketio, db
from app.models.ai import PredictionLog

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
        """Guarda una predicción en la base de datos."""
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
                    'network_latency': kiosk_data.get('network_latency')
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