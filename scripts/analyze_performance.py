#!/usr/bin/env python
# Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

"""
Script para analizar el rendimiento del sistema.

Funcionalidad:
- Analiza logs de rendimiento y métricas
- Identifica patrones de uso y cuellos de botella
- Genera gráficos y reportes de rendimiento
- Calcula estadísticas de uso de recursos
- Proporciona recomendaciones de optimización

Uso:
python scripts/analyze_performance.py [--days N]

Argumentos:
--days: Número de días a analizar (default: 7)

Salida:
- Genera reportes en formato HTML y PDF
- Crea gráficos de tendencias
- Exporta datos en CSV para análisis adicional
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from app.services.monitoring_service import MonitoringService
from app.services.notification_service import NotificationService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Analiza resultados de pruebas de carga')
    parser.add_argument('--threshold', type=int, default=1000,
                       help='Umbral máximo de tiempo de respuesta en ms')
    parser.add_argument('--results-dir', type=str, default='tests/load/results',
                       help='Directorio con resultados de pruebas')
    return parser.parse_args()

def analyze_results(results_dir: str, threshold: int) -> Dict:
    """Analiza los resultados de las pruebas de carga."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"No se encontró el directorio {results_dir}")
    
    # Buscar el archivo de resultados más reciente
    json_files = list(results_path.glob('*.json'))
    if not json_files:
        raise FileNotFoundError("No se encontraron archivos de resultados")
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file) as f:
        data = json.load(f)
    
    # Analizar métricas
    metrics = {
        'total_requests': data.get('total_requests', 0),
        'failed_requests': data.get('failed_requests', 0),
        'average_response_time': data.get('average_response_time', 0),
        'p95_response_time': data.get('p95_response_time', 0),
        'test_duration': data.get('duration', 0)
    }
    
    # Calcular tasa de error
    metrics['error_rate'] = (metrics['failed_requests'] / metrics['total_requests'] * 100 
                           if metrics['total_requests'] > 0 else 0)
    
    # Verificar umbrales
    metrics['threshold_exceeded'] = metrics['p95_response_time'] > threshold
    
    return metrics

def main():
    """Función principal."""
    args = parse_args()
    
    try:
        # Analizar resultados
        metrics = analyze_results(args.results_dir, args.threshold)
        
        # Registrar métricas en el sistema de monitoreo
        monitoring = MonitoringService()
        monitoring.register_load_test_metrics(metrics)
        
        # Verificar si se excedieron los umbrales
        if metrics['threshold_exceeded']:
            logger.warning(f"Se excedió el umbral de tiempo de respuesta: {metrics['p95_response_time']}ms > {args.threshold}ms")
            
            # Enviar notificación
            notification = NotificationService()
            notification.send_alert(
                title="Alerta de Rendimiento",
                message=f"Las pruebas de carga excedieron el umbral de tiempo de respuesta.\n"
                       f"P95: {metrics['p95_response_time']}ms (umbral: {args.threshold}ms)\n"
                       f"Tasa de error: {metrics['error_rate']}%",
                severity="warning"
            )
            
            # Fallar el build si la tasa de error es muy alta
            if metrics['error_rate'] > 5:
                logger.error(f"Tasa de error demasiado alta: {metrics['error_rate']}%")
                exit(1)
        
        logger.info("Análisis de rendimiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el análisis de rendimiento: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main() 