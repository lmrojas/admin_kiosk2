"""
Script para el pipeline de reentrenamiento automático de modelos.

Funcionalidad:
- Monitorea rendimiento de modelos en producción
- Detecta degradación del rendimiento
- Recolecta nuevos datos de entrenamiento
- Ejecuta reentrenamiento automático
- Valida y despliega nuevos modelos

Uso:
python scripts/auto_retrain_pipeline.py [--threshold SCORE]

Argumentos:
--threshold: Umbral de rendimiento para reentrenamiento
--force: Forzar reentrenamiento
--dry-run: Simular sin aplicar cambios

Notas:
- Se ejecuta periódicamente vía scheduler
- Requiere acceso a datos de producción
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import os
import sys
import logging
from datetime import datetime, timedelta
import torch
import numpy as np
from pathlib import Path

# Agregar directorio raíz al path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(root_dir)

from app import db, create_app
from app.models.kiosk import SensorData
from app.services.kiosk_ai_service import KioskAIService, SimpleAnomalyDetector

class AutoRetrainPipeline:
    """Pipeline para reentrenamiento automático del modelo de IA"""

    def __init__(self, days_of_data=30, min_samples=1000):
        """
        Inicializa el pipeline de reentrenamiento.
        
        Args:
            days_of_data (int): Días de datos históricos a usar
            min_samples (int): Mínimo de muestras requeridas
        """
        self.days_of_data = days_of_data
        self.min_samples = min_samples
        self.ai_service = KioskAIService()
        self.logger = logging.getLogger(__name__)

    def extract_real_data(self):
        """
        Extrae datos reales de la base de datos.
        
        Returns:
            list: Lista de diccionarios con datos de sensores
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.days_of_data)
        sensor_data = SensorData.query.filter(
            SensorData.timestamp >= cutoff_date
        ).all()

        return [{
            'cpu_usage': data.cpu_usage,
            'memory_usage': data.memory_usage,
            'network_latency': data.network_latency,
            'is_anomaly': 1 if data.kiosk.anomaly_probability > 0.5 else 0
        } for data in sensor_data]

    def evaluate_model(self, model, test_data):
        """
        Evalúa el rendimiento del modelo en datos de prueba.
        
        Args:
            model: Modelo de PyTorch
            test_data (list): Datos de prueba
        
        Returns:
            dict: Métricas de evaluación
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in test_data:
                features = torch.tensor([
                    data['cpu_usage'] / 100.0,
                    data['memory_usage'] / 100.0,
                    min(data['network_latency'] / 300.0, 1.0)
                ], dtype=torch.float32).unsqueeze(0)
                
                output = model(features)
                predicted = (output > 0.5).float()
                label = torch.tensor([[data['is_anomaly']]], dtype=torch.float32)
                
                correct += (predicted == label).sum().item()
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy}

    def run_pipeline(self):
        """
        Ejecuta el pipeline completo de reentrenamiento.
        
        Returns:
            bool: True si el reentrenamiento fue exitoso
        """
        try:
            # 1. Extraer datos reales
            real_data = self.extract_real_data()
            self.logger.info(f"Datos reales extraídos: {len(real_data)} muestras")

            # 2. Generar datos sintéticos si es necesario
            if len(real_data) < self.min_samples:
                synthetic_data = self.ai_service.generate_synthetic_data(
                    num_samples=self.min_samples - len(real_data)
                )
                training_data = real_data + synthetic_data
                self.logger.info(f"Datos sintéticos generados: {len(synthetic_data)} muestras")
            else:
                training_data = real_data

            # 3. Dividir datos en entrenamiento y prueba
            np.random.shuffle(training_data)
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            test_data = training_data[split_idx:]

            # 4. Crear y entrenar nuevo modelo
            model = SimpleAnomalyDetector()
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters())

            # 5. Entrenamiento
            model.train()
            for epoch in range(100):  # 100 épocas
                total_loss = 0
                for data in train_data:
                    features = torch.tensor([
                        data['cpu_usage'] / 100.0,
                        data['memory_usage'] / 100.0,
                        min(data['network_latency'] / 300.0, 1.0)
                    ], dtype=torch.float32).unsqueeze(0)
                    
                    label = torch.tensor([[data['is_anomaly']]], dtype=torch.float32)
                    
                    optimizer.zero_grad()
                    output = model(features)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_data)
                if epoch % 10 == 0:
                    self.logger.info(f"Época {epoch}, Loss: {avg_loss:.4f}")

            # 6. Evaluar modelo
            metrics = self.evaluate_model(model, test_data)
            self.logger.info(f"Métricas de evaluación: {metrics}")

            # 7. Guardar modelo si supera umbral de precisión
            if metrics['accuracy'] > 0.8:  # 80% de precisión mínima
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = os.path.join(root_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
                
                model_path = os.path.join(model_dir, f'kiosk_anomaly_model_{timestamp}.pth')
                torch.save(model, model_path)
                self.logger.info(f"Modelo guardado en: {model_path}")
                return True
            else:
                self.logger.warning("Modelo no alcanzó precisión mínima requerida")
                return False

        except Exception as e:
            self.logger.error(f"Error en pipeline de reentrenamiento: {str(e)}")
            return False

def main():
    """Función principal para ejecutar el pipeline"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Crear aplicación Flask y contexto
    app = create_app()
    with app.app_context():
        pipeline = AutoRetrainPipeline()
        success = pipeline.run_pipeline()
        
        if success:
            logger.info("Pipeline de reentrenamiento completado exitosamente")
        else:
            logger.error("Pipeline de reentrenamiento falló")

if __name__ == '__main__':
    main() 