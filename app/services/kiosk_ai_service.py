# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union

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
    """
    Servicio de IA para predicción de anomalías en kiosks.
    Sigue el patrón de Services, separando la lógica de negocio.
    """

    def __init__(self, model_path='models/kiosk_anomaly_model.pth'):
        """
        Inicializa el servicio de IA cargando un modelo pre-entrenado.
        
        Args:
            model_path (str): Ruta al modelo de IA pre-entrenado
        """
        self.model = self._load_model(model_path)

    def _load_model(self, path):
        """
        Carga un modelo de IA desde un archivo.
        
        Args:
            path (str): Ruta al archivo del modelo
        
        Returns:
            torch.nn.Module: Modelo de IA cargado
        """
        try:
            model = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
            model.eval()  # Modo de evaluación
            return model
        except FileNotFoundError:
            print(f"[WARN] Modelo no encontrado en {path}. Creando modelo por defecto.")
            return SimpleAnomalyDetector()
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo: {e}")
            return SimpleAnomalyDetector()

    def predict_anomaly(self, kiosk_data: Dict[str, Union[float, int]]) -> float:
        """
        Predice la probabilidad de anomalía para un kiosk.
        
        Args:
            kiosk_data (dict): Métricas del kiosk
                - cpu_usage (float): Uso de CPU
                - memory_usage (float): Uso de memoria
                - network_latency (float): Latencia de red
        
        Returns:
            float: Probabilidad de anomalía (0.0 - 1.0)
        """
        # Validar y normalizar datos
        features = [
            kiosk_data.get('cpu_usage', 0.0) / 100.0,
            kiosk_data.get('memory_usage', 0.0) / 100.0,
            min(kiosk_data.get('network_latency', 0.0) / 300.0, 1.0)
        ]

        # Convertir a tensor
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Predecir
        with torch.no_grad():
            anomaly_prob = self.model(input_tensor).item()

        return anomaly_prob

    def train_model(self, training_data):
        """
        Método para reentrenar el modelo con nuevos datos.
        NOTA: La implementación completa de reentrenamiento 
        se hará en scripts separados.
        
        Args:
            training_data (list): Datos de entrenamiento
        """
        # TODO: Implementar lógica completa de reentrenamiento
        pass

    def generate_synthetic_data(self, num_samples=1000):
        """
        Genera datos sintéticos para entrenamiento.
        
        Args:
            num_samples (int): Número de muestras a generar
        
        Returns:
            list: Datos sintéticos de kiosks
        """
        synthetic_data = []
        for _ in range(num_samples):
            cpu_usage = np.random.uniform(0, 100)
            memory_usage = np.random.uniform(0, 100)
            network_latency = np.random.uniform(10, 300)
            
            # Heurística simple para etiquetar anomalías
            is_anomaly = 1 if (cpu_usage > 90 or memory_usage > 90 or network_latency > 250) else 0
            
            synthetic_data.append({
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'network_latency': network_latency,
                'is_anomaly': is_anomaly
            })
        
        return synthetic_data 