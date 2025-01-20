"""
Script para entrenar el modelo de IA del sistema.

Funcionalidad:
- Entrena modelos de detección de anomalías
- Procesa datos históricos de kiosks
- Optimiza hiperparámetros del modelo
- Evalúa rendimiento del modelo
- Guarda checkpoints del entrenamiento

Uso:
python scripts/train_ai_model.py [--epochs N] [--batch-size N]

Argumentos:
--epochs: Número de épocas de entrenamiento (default: 100)
--batch-size: Tamaño del batch (default: 32)
--data-dir: Directorio con datos de entrenamiento

Salida:
- Modelo entrenado (.pth)
- Métricas de entrenamiento
- Gráficos de convergencia
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import os
import sys
import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Agregar directorio raíz al path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(root_dir)

from app.services.kiosk_ai_service import SimpleAnomalyDetector
from scripts.generate_synthetic_data import SyntheticDataGenerator

class ModelTrainer:
    """Entrenador del modelo de detección de anomalías"""

    def __init__(self, model_dir='models'):
        """
        Inicializa el entrenador.
        
        Args:
            model_dir (str): Directorio donde guardar los modelos
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()

    def prepare_data(self, real_data_path=None, synthetic_samples=1000):
        """
        Prepara los datos para entrenamiento.
        
        Args:
            real_data_path (str): Ruta al archivo CSV con datos reales
            synthetic_samples (int): Número de muestras sintéticas a generar
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        data = []
        
        # Cargar datos reales si existen
        if real_data_path and os.path.exists(real_data_path):
            real_df = pd.read_csv(real_data_path)
            real_data = real_df[['cpu_usage', 'memory_usage', 'network_latency']].values
            real_labels = real_df['is_anomaly'].values
            self.logger.info(f"Datos reales cargados: {len(real_df)} muestras")
            
            # Normalizar datos reales
            data.append(real_data)
        
        # Generar datos sintéticos
        generator = SyntheticDataGenerator()
        synthetic_df = generator.generate_dataset(synthetic_samples)
        synthetic_data = synthetic_df[['cpu_usage', 'memory_usage', 'network_latency']].values
        synthetic_labels = synthetic_df['is_anomaly'].values
        self.logger.info(f"Datos sintéticos generados: {len(synthetic_df)} muestras")
        
        data.append(synthetic_data)
        
        # Combinar y normalizar datos
        X = np.vstack(data)
        X = self.scaler.fit_transform(X)
        
        # Combinar etiquetas
        y = np.concatenate([real_labels, synthetic_labels]) if real_data_path else synthetic_labels
        
        # Dividir en train/test
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X_train (np.array): Datos de entrenamiento
            y_train (np.array): Etiquetas de entrenamiento
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
            learning_rate (float): Tasa de aprendizaje
        
        Returns:
            SimpleAnomalyDetector: Modelo entrenado
        """
        # Crear modelo
        model = SimpleAnomalyDetector()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Convertir datos a tensores
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        
        # Entrenamiento
        model.train()
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            if epoch % 10 == 0:
                self.logger.info(f"Época {epoch}, Loss: {avg_loss:.4f}")
        
        return model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evalúa el modelo en datos de prueba.
        
        Args:
            model: Modelo entrenado
            X_test (np.array): Datos de prueba
            y_test (np.array): Etiquetas de prueba
        
        Returns:
            dict: Métricas de evaluación
        """
        model.eval()
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).reshape(-1, 1)
        
        with torch.no_grad():
            outputs = model(X_test)
            predictions = (outputs >= 0.5).float()
            
            correct = (predictions == y_test).sum().item()
            total = len(y_test)
            accuracy = correct / total
            
            # Calcular métricas adicionales
            tp = ((predictions == 1) & (y_test == 1)).sum().item()
            fp = ((predictions == 1) & (y_test == 0)).sum().item()
            fn = ((predictions == 0) & (y_test == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def save_model(self, model, metrics):
        """
        Guarda el modelo entrenado.
        
        Args:
            model: Modelo a guardar
            metrics (dict): Métricas del modelo
        
        Returns:
            str: Ruta donde se guardó el modelo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f'kiosk_anomaly_model_{timestamp}.pth'
        
        # Guardar modelo
        torch.save(model, model_path)
        
        # Guardar métricas
        metrics_path = self.model_dir / f'metrics_{timestamp}.txt'
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        self.logger.info(f"Modelo guardado en: {model_path}")
        self.logger.info(f"Métricas guardadas en: {metrics_path}")
        
        return str(model_path)

def main():
    """Función principal para entrenar el modelo"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Crear entrenador
        trainer = ModelTrainer()
        
        # Preparar datos
        real_data_path = os.path.join(root_dir, 'data', 'real', 'sensor_data.csv')
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            real_data_path=real_data_path if os.path.exists(real_data_path) else None,
            synthetic_samples=10000
        )
        
        # Entrenar modelo
        model = trainer.train_model(X_train, y_train)
        
        # Evaluar modelo
        metrics = trainer.evaluate_model(model, X_test, y_test)
        logger.info(f"Métricas de evaluación: {metrics}")
        
        # Guardar modelo si cumple criterios mínimos
        if metrics['f1_score'] > 0.8:
            trainer.save_model(model, metrics)
            logger.info("Entrenamiento completado exitosamente")
        else:
            logger.warning("Modelo no alcanzó métricas mínimas requeridas")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 