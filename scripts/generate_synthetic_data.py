"""
Script para generar datos sintéticos para pruebas y desarrollo.

Funcionalidad:
- Genera datos realistas de kiosks
- Simula patrones de uso y comportamiento
- Crea anomalías controladas para testing
- Genera series temporales de métricas
- Exporta datos en múltiples formatos

Uso:
python scripts/generate_synthetic_data.py [--samples N] [--anomalies]

Argumentos:
--samples: Número de muestras a generar (default: 1000)
--anomalies: Incluir anomalías en los datos
--output: Formato de salida (csv/json)

Notas:
- Solo para desarrollo y pruebas
- No usar en producción
"""

# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN @cura.md
# Y @project_custom_structure.txt

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Agregar directorio raíz al path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(root_dir)

from app.services.kiosk_ai_service import KioskAIService

class SyntheticDataGenerator:
    """Generador de datos sintéticos para entrenamiento del modelo de IA"""

    def __init__(self):
        """Inicializa el generador de datos sintéticos"""
        self.logger = logging.getLogger(__name__)

    def generate_normal_pattern(self, num_samples):
        """
        Genera patrones normales de uso de kiosks.
        
        Args:
            num_samples (int): Número de muestras a generar
        
        Returns:
            list: Lista de diccionarios con datos sintéticos normales
        """
        data = []
        for _ in range(num_samples):
            # Generar métricas normales
            cpu_usage = np.random.normal(50, 15)  # Media 50%, std 15%
            memory_usage = np.random.normal(60, 10)  # Media 60%, std 10%
            network_latency = np.random.normal(100, 30)  # Media 100ms, std 30ms
            
            # Ajustar valores a rangos válidos
            cpu_usage = np.clip(cpu_usage, 0, 100)
            memory_usage = np.clip(memory_usage, 0, 100)
            network_latency = max(0, network_latency)
            
            data.append({
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'network_latency': network_latency,
                'is_anomaly': 0
            })
        
        return data

    def generate_anomaly_pattern(self, num_samples):
        """
        Genera patrones anómalos de uso de kiosks.
        
        Args:
            num_samples (int): Número de muestras a generar
        
        Returns:
            list: Lista de diccionarios con datos sintéticos anómalos
        """
        data = []
        anomaly_types = [
            'high_cpu',
            'high_memory',
            'high_latency',
            'combined'
        ]
        
        for _ in range(num_samples):
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'high_cpu':
                cpu_usage = np.random.uniform(90, 100)
                memory_usage = np.random.normal(60, 10)
                network_latency = np.random.normal(100, 30)
            
            elif anomaly_type == 'high_memory':
                cpu_usage = np.random.normal(50, 15)
                memory_usage = np.random.uniform(90, 100)
                network_latency = np.random.normal(100, 30)
            
            elif anomaly_type == 'high_latency':
                cpu_usage = np.random.normal(50, 15)
                memory_usage = np.random.normal(60, 10)
                network_latency = np.random.uniform(250, 500)
            
            else:  # combined
                cpu_usage = np.random.uniform(85, 100)
                memory_usage = np.random.uniform(85, 100)
                network_latency = np.random.uniform(200, 400)
            
            # Ajustar valores a rangos válidos
            cpu_usage = np.clip(cpu_usage, 0, 100)
            memory_usage = np.clip(memory_usage, 0, 100)
            network_latency = max(0, network_latency)
            
            data.append({
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'network_latency': network_latency,
                'is_anomaly': 1
            })
        
        return data

    def generate_dataset(self, total_samples, anomaly_ratio=0.2):
        """
        Genera un conjunto completo de datos sintéticos.
        
        Args:
            total_samples (int): Número total de muestras
            anomaly_ratio (float): Proporción de anomalías en el dataset
        
        Returns:
            pd.DataFrame: Dataset completo con datos sintéticos
        """
        num_anomalies = int(total_samples * anomaly_ratio)
        num_normal = total_samples - num_anomalies
        
        self.logger.info(f"Generando {num_normal} muestras normales y {num_anomalies} anomalías")
        
        # Generar datos
        normal_data = self.generate_normal_pattern(num_normal)
        anomaly_data = self.generate_anomaly_pattern(num_anomalies)
        
        # Combinar datos
        all_data = normal_data + anomaly_data
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_data)
        
        # Agregar timestamps
        now = datetime.utcnow()
        timestamps = [now - timedelta(minutes=i) for i in range(len(df))]
        df['timestamp'] = timestamps
        
        # Reordenar columnas
        df = df[['timestamp', 'cpu_usage', 'memory_usage', 'network_latency', 'is_anomaly']]
        
        return df

def main():
    """Función principal para generar datos sintéticos"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Crear generador
        generator = SyntheticDataGenerator()
        
        # Generar dataset
        total_samples = 10000  # 10k muestras
        df = generator.generate_dataset(total_samples)
        
        # Guardar dataset
        output_dir = Path(root_dir) / 'data' / 'synthetic'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'synthetic_data_{timestamp}.csv'
        
        df.to_csv(output_file, index=False)
        logger.info(f"Dataset guardado en: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generando datos sintéticos: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 