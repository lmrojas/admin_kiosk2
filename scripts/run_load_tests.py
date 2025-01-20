#!/usr/bin/env python
# Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

"""
Script para ejecutar pruebas de carga en el sistema.

Funcionalidad:
- Simula múltiples usuarios concurrentes
- Prueba diferentes endpoints de la API
- Mide tiempos de respuesta y rendimiento
- Genera reportes detallados de resultados
- Detecta cuellos de botella y límites del sistema

Uso:
python scripts/run_load_tests.py [--users N] [--time T]

Argumentos:
--users: Número de usuarios concurrentes (default: 100)
--time: Duración de la prueba en segundos (default: 300)

Notas:
- Requiere que el sistema esté en ejecución
- No ejecutar en producción sin precaución
"""

import os
import json
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/load_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Ejecutar pruebas de carga y rendimiento')
    parser.add_argument('--users', type=int, default=100,
                      help='Número de usuarios concurrentes')
    parser.add_argument('--spawn-rate', type=int, default=10,
                      help='Tasa de generación de usuarios por segundo')
    parser.add_argument('--time', type=str, default='5m',
                      help='Duración de la prueba (ej: 5m, 1h)')
    parser.add_argument('--host', type=str, default='http://localhost:5000',
                      help='Host de la aplicación')
    parser.add_argument('--db-tests', action='store_true',
                      help='Ejecutar también tests de rendimiento de BD')
    parser.add_argument('--export-graphs', action='store_true',
                      help='Exportar gráficos de resultados')
    return parser.parse_args()

def run_load_test(args) -> Dict:
    """Ejecutar pruebas de carga con Locust"""
    try:
        # Crear directorio de resultados si no existe
        results_dir = Path('tests/load/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombres de archivos con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_report = results_dir / f'report_{timestamp}.html'
        json_metrics = results_dir / f'metrics_{timestamp}.json'
        
        # Construir comando de Locust
        cmd = [
            'locust',
            '-f', 'tests/load/test_load.py',
            '--headless',
            '--users', str(args.users),
            '--spawn-rate', str(args.spawn_rate),
            '--run-time', args.time,
            '--host', args.host,
            '--html', str(html_report),
            '--json', str(json_metrics)
        ]
        
        # Ejecutar Locust
        logger.info(f'Iniciando pruebas de carga con {args.users} usuarios')
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            logger.info('Pruebas de carga completadas exitosamente')
            with open(json_metrics) as f:
                metrics = json.load(f)
            return metrics
        else:
            logger.error(f'Error en pruebas de carga: {process.stderr}')
            return None
            
    except Exception as e:
        logger.error(f'Error ejecutando pruebas de carga: {str(e)}')
        return None

def run_db_performance_tests() -> Dict:
    """Ejecutar pruebas de rendimiento de base de datos"""
    try:
        logger.info('Iniciando pruebas de rendimiento de BD')
        cmd = [
            'pytest',
            'tests/performance/test_db_optimizations.py',
            '-v',
            '--benchmark-only',
            '--benchmark-json=tests/performance/results/db_benchmark.json'
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            logger.info('Pruebas de rendimiento de BD completadas exitosamente')
            with open('tests/performance/results/db_benchmark.json') as f:
                return json.load(f)
        else:
            logger.error(f'Error en pruebas de BD: {process.stderr}')
            return None
            
    except Exception as e:
        logger.error(f'Error ejecutando pruebas de BD: {str(e)}')
        return None

def analyze_results(load_metrics: Dict, db_metrics: Dict = None) -> Dict:
    """Analizar resultados de las pruebas"""
    analysis = {
        'load_test': {
            'total_requests': load_metrics.get('total_requests', 0),
            'failed_requests': load_metrics.get('failed_requests', 0),
            'average_response_time': load_metrics.get('average_response_time', 0),
            'percentile_95': load_metrics.get('percentile_95', 0),
            'requests_per_second': load_metrics.get('requests_per_second', 0)
        }
    }
    
    if db_metrics:
        analysis['db_performance'] = {
            test['name']: {
                'min': test['stats']['min'],
                'max': test['stats']['max'],
                'mean': test['stats']['mean'],
                'median': test['stats']['median'],
                'stddev': test['stats']['stddev']
            }
            for test in db_metrics.get('benchmarks', [])
        }
    
    return analysis

def generate_graphs(analysis: Dict, output_dir: Path):
    """Generar gráficos de resultados"""
    # Gráfico de tiempos de respuesta
    plt.figure(figsize=(10, 6))
    metrics = ['average_response_time', 'percentile_95']
    values = [analysis['load_test'][m] for m in metrics]
    plt.bar(metrics, values)
    plt.title('Tiempos de Respuesta (ms)')
    plt.savefig(output_dir / 'response_times.png')
    plt.close()
    
    if 'db_performance' in analysis:
        # Gráfico de rendimiento de BD
        plt.figure(figsize=(12, 6))
        tests = list(analysis['db_performance'].keys())
        means = [analysis['db_performance'][t]['mean'] for t in tests]
        plt.barh(tests, means)
        plt.title('Tiempo Medio de Ejecución por Test (ms)')
        plt.tight_layout()
        plt.savefig(output_dir / 'db_performance.png')
        plt.close()

def main():
    """Función principal"""
    args = parse_args()
    
    try:
        # Ejecutar pruebas de carga
        load_metrics = run_load_test(args)
        if not load_metrics:
            return 1
        
        # Ejecutar pruebas de BD si se solicitó
        db_metrics = None
        if args.db_tests:
            db_metrics = run_db_performance_tests()
        
        # Analizar resultados
        analysis = analyze_results(load_metrics, db_metrics)
        
        # Guardar resultados
        results_dir = Path('tests/load/results')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(results_dir / f'analysis_{timestamp}.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generar gráficos si se solicitó
        if args.export_graphs:
            generate_graphs(analysis, results_dir)
        
        logger.info('Análisis de pruebas completado exitosamente')
        return 0
        
    except Exception as e:
        logger.error(f'Error durante la ejecución: {str(e)}')
        return 1

if __name__ == '__main__':
    exit(main()) 