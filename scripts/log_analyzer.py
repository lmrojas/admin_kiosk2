# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict
import argparse
import logging

class LogAnalyzer:
    """
    Analizador de logs para el sistema de administración de kiosks
    """

    def __init__(self, log_directory='app/logs'):
        """
        Inicializar el analizador de logs
        
        Args:
            log_directory (str): Directorio de archivos de log
        """
        self.log_directory = log_directory
        self.log_pattern = re.compile(
            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+) - (\w+) - \[([\w\.]+):(\d+)\] - (.+)$'
        )

    def _read_log_files(self, days=1):
        """
        Leer archivos de log de los últimos días
        
        Args:
            days (int): Número de días de logs a analizar
        
        Returns:
            list: Líneas de log
        """
        log_files = []
        current_date = datetime.now()
        
        for i in range(days):
            log_date = current_date - timedelta(days=i)
            log_filename = f'admin_kiosk_{log_date.strftime("%Y-%m-%d")}.log'
            log_path = os.path.join(self.log_directory, log_filename)
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as log_file:
                    log_files.extend(log_file.readlines())
        
        return log_files

    def analyze_authentication_events(self, days=1):
        """
        Analizar eventos de autenticación
        
        Args:
            days (int): Número de días de logs a analizar
        
        Returns:
            dict: Resumen de eventos de autenticación
        """
        log_lines = self._read_log_files(days)
        auth_events = {
            'login_success': 0,
            'login_failed': 0,
            'registrations': 0,
            'password_changes': 0
        }
        
        for line in log_lines:
            match = self.log_pattern.match(line.strip())
            if match and 'admin_kiosk.auth' in match.group(2):
                message = match.group(6)
                
                if 'login - Usuario' in message and 'Estado: Fallido' in message:
                    auth_events['login_failed'] += 1
                elif 'login - Usuario' in message and 'Estado: Fallido' not in message:
                    auth_events['login_success'] += 1
                elif 'registro - Usuario' in message:
                    auth_events['registrations'] += 1
                elif 'cambio_contraseña - Usuario' in message:
                    auth_events['password_changes'] += 1
        
        return auth_events

    def analyze_kiosk_events(self, days=1):
        """
        Analizar eventos de kiosks
        
        Args:
            days (int): Número de días de logs a analizar
        
        Returns:
            dict: Resumen de eventos de kiosks
        """
        log_lines = self._read_log_files(days)
        kiosk_events = defaultdict(int)
        
        for line in log_lines:
            match = self.log_pattern.match(line.strip())
            if match and 'admin_kiosk.kiosk' in match.group(2):
                message = match.group(6)
                
                if 'Evento de Kiosk:' in message:
                    event_type = message.split('Evento de Kiosk:')[1].split('-')[0].strip()
                    kiosk_events[event_type] += 1
        
        return dict(kiosk_events)

    def analyze_system_errors(self, days=1):
        """
        Analizar errores del sistema
        
        Args:
            days (int): Número de días de logs a analizar
        
        Returns:
            dict: Resumen de errores del sistema
        """
        log_lines = self._read_log_files(days)
        system_errors = defaultdict(int)
        
        for line in log_lines:
            match = self.log_pattern.match(line.strip())
            if match and 'admin_kiosk.errors' in match.group(2):
                message = match.group(6)
                
                if 'Error del Sistema:' in message:
                    error_type = message.split('Error del Sistema:')[1].split('-')[0].strip()
                    system_errors[error_type] += 1
        
        return dict(system_errors)

    def generate_report(self, days=1):
        """
        Generar informe de análisis de logs
        
        Args:
            days (int): Número de días de logs a analizar
        
        Returns:
            dict: Informe completo de análisis de logs
        """
        report = {
            'fecha_generacion': datetime.now().isoformat(),
            'dias_analizados': days,
            'eventos_autenticacion': self.analyze_authentication_events(days),
            'eventos_kiosks': self.analyze_kiosk_events(days),
            'errores_sistema': self.analyze_system_errors(days)
        }
        
        return report

    def save_report(self, report, output_file=None):
        """
        Guardar informe en archivo JSON
        
        Args:
            report (dict): Informe a guardar
            output_file (str, opcional): Ruta del archivo de salida
        
        Returns:
            str: Ruta del archivo de salida
        """
        if not output_file:
            output_file = f'log_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_file

def main():
    """
    Función principal para ejecutar el análisis de logs desde la línea de comandos
    """
    parser = argparse.ArgumentParser(description='Analizador de logs del sistema de administración de kiosks')
    parser.add_argument('-d', '--days', type=int, default=1, 
                        help='Número de días de logs a analizar')
    parser.add_argument('-o', '--output', type=str, 
                        help='Archivo de salida para el informe')
    
    args = parser.parse_args()
    
    try:
        analyzer = LogAnalyzer()
        report = analyzer.generate_report(days=args.days)
        output_file = analyzer.save_report(report, args.output)
        
        print(f"Informe generado: {output_file}")
        print("\nResumen:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    except Exception as e:
        logging.error(f"Error al generar el informe de logs: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 