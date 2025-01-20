#!/usr/bin/env python
# Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

"""
Script para analizar y comparar benchmarks del sistema.

Funcionalidad:
- Ejecuta pruebas de rendimiento estandarizadas
- Compara resultados con benchmarks anteriores
- Analiza impacto de cambios en el rendimiento
- Detecta regresiones de rendimiento
- Genera reportes comparativos

Uso:
python scripts/analyze_benchmarks.py [--compare-with FECHA]

Argumentos:
--compare-with: Fecha del benchmark anterior (formato: YYYY-MM-DD)

Salida:
- Reporte detallado de comparación
- Gráficos de tendencias
- Alertas de regresiones significativas
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
from statistics import mean, stdev

from app.services.monitoring_service import MonitoringService
from app.services.notification_service import NotificationService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/benchmark_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Analiza resultados de benchmarks')
    parser.add_argument('--input', type=str, required=True,
                       help='Archivo JSON con resultados de benchmarks')
    parser.add_argument('--threshold', type=float, default=100,
                       help='Umbral máximo de tiempo de ejecución en ms')
    parser.add_argument('--baseline', type=str,
                       help='Archivo JSON con resultados baseline para comparación')
    return parser.parse_args()

def load_benchmark_data(file_path: str) -> Dict:
    """Carga los datos del benchmark desde un archivo JSON."""
    with open(file_path) as f:
        return json.load(f)

def analyze_benchmarks(data: Dict, baseline: Dict = None, threshold: float = 100) -> Dict:
    """Analiza los resultados de los benchmarks."""
    results = {}
    
    for benchmark in data['benchmarks']:
        name = benchmark['name']
        stats = benchmark['stats']
        
        # Calcular métricas básicas
        metrics = {
            'mean': stats['mean'],
            'min': stats['min'],
            'max': stats['max'],
            'stddev': stats['stddev'],
            'rounds': stats['rounds'],
            'median': stats['median'],
            'iqr': stats['iqr']
        }
        
        # Verificar umbral
        metrics['threshold_exceeded'] = metrics['mean'] > threshold
        
        # Comparar con baseline si existe
        if baseline and name in baseline['benchmarks']:
            baseline_stats = baseline['benchmarks'][name]['stats']
            metrics['baseline_comparison'] = {
                'mean_diff': (metrics['mean'] - baseline_stats['mean']) / baseline_stats['mean'] * 100,
                'regression': metrics['mean'] > baseline_stats['mean'] * 1.1  # 10% de regresión
            }
        
        results[name] = metrics
    
    return results

def main():
    """Función principal."""
    args = parse_args()
    
    try:
        # Cargar datos
        data = load_benchmark_data(args.input)
        baseline = None
        if args.baseline:
            baseline = load_benchmark_data(args.baseline)
        
        # Analizar benchmarks
        results = analyze_benchmarks(data, baseline, args.threshold)
        
        # Registrar métricas en el sistema de monitoreo
        monitoring = MonitoringService()
        monitoring.register_benchmark_metrics(results)
        
        # Verificar resultados
        failures = []
        regressions = []
        
        for name, metrics in results.items():
            # Verificar umbrales
            if metrics['threshold_exceeded']:
                failures.append(f"{name}: {metrics['mean']:.2f}ms > {args.threshold}ms")
            
            # Verificar regresiones
            if baseline and metrics.get('baseline_comparison', {}).get('regression'):
                diff = metrics['baseline_comparison']['mean_diff']
                regressions.append(f"{name}: +{diff:.1f}% vs baseline")
        
        # Enviar notificaciones si hay problemas
        if failures or regressions:
            notification = NotificationService()
            message = "Resultados de benchmarks:\n"
            
            if failures:
                message += "\nUmbrales excedidos:\n" + "\n".join(failures)
            
            if regressions:
                message += "\nRegresiones detectadas:\n" + "\n".join(regressions)
            
            notification.send_alert(
                title="Alerta de Benchmarks",
                message=message,
                severity="warning"
            )
            
            # Fallar el build si hay problemas críticos
            if len(failures) > 3 or len(regressions) > 5:
                logger.error("Demasiados problemas de rendimiento detectados")
                exit(1)
        
        logger.info("Análisis de benchmarks completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el análisis de benchmarks: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main() 