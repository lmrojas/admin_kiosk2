# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import os
import sys
import json
import argparse
import requests
import logging
from datetime import datetime, timedelta
from app import create_app, db
from app.models.kiosk import Kiosk
from app.services.kiosk_service import KioskService
from config.logging_config import LoggingConfig

class KioskMonitor:
    """
    Clase para monitorear el estado de los kiosks
    """

    def __init__(self, app=None):
        """
        Inicializar monitor de kiosks
        
        Args:
            app (Flask, opcional): Instancia de aplicación Flask
        """
        self.app = app or create_app()
        self.logger = logging.getLogger('admin_kiosk.monitor')

    def check_kiosk_health(self, kiosk_uuid):
        """
        Verificar el estado de salud de un kiosk
        
        Args:
            kiosk_uuid (str): UUID del kiosk
        
        Returns:
            dict: Estado de salud del kiosk
        """
        with self.app.app_context():
            kiosk = KioskService.get_kiosk_by_uuid(kiosk_uuid)
            
            if not kiosk:
                self.logger.error(f"Kiosk con UUID {kiosk_uuid} no encontrado")
                return {
                    'status': 'error',
                    'message': 'Kiosk no encontrado'
                }
            
            health_status = {
                'uuid': kiosk.uuid,
                'name': kiosk.name,
                'status': kiosk.status,
                'last_online': kiosk.last_online.isoformat() if kiosk.last_online else None,
                'hardware': {
                    'cpu_model': kiosk.cpu_model,
                    'ram_total': kiosk.ram_total,
                    'storage_total': kiosk.storage_total
                },
                'network': {
                    'ip_address': kiosk.ip_address,
                    'mac_address': kiosk.mac_address
                }
            }
            
            return health_status

    def check_all_kiosks_health(self):
        """
        Verificar el estado de salud de todos los kiosks
        
        Returns:
            list: Lista de estados de salud de kiosks
        """
        with self.app.app_context():
            kiosks = Kiosk.query.all()
            health_statuses = []
            
            for kiosk in kiosks:
                health_status = self.check_kiosk_health(kiosk.uuid)
                health_statuses.append(health_status)
            
            return health_statuses

    def detect_inactive_kiosks(self, inactivity_threshold_days=7):
        """
        Detectar kiosks inactivos
        
        Args:
            inactivity_threshold_days (int): Días de inactividad para considerar un kiosk inactivo
        
        Returns:
            list: Lista de kiosks inactivos
        """
        with self.app.app_context():
            threshold_date = datetime.utcnow() - timedelta(days=inactivity_threshold_days)
            
            inactive_kiosks = Kiosk.query.filter(
                (Kiosk.last_online < threshold_date) | (Kiosk.last_online == None)
            ).all()
            
            inactive_list = []
            for kiosk in inactive_kiosks:
                inactive_list.append({
                    'uuid': kiosk.uuid,
                    'name': kiosk.name,
                    'last_online': kiosk.last_online.isoformat() if kiosk.last_online else None
                })
            
            return inactive_list

    def generate_kiosk_report(self, output_file=None):
        """
        Generar informe de estado de kiosks
        
        Args:
            output_file (str, opcional): Ruta del archivo de salida
        
        Returns:
            dict: Informe de estado de kiosks
        """
        health_statuses = self.check_all_kiosks_health()
        inactive_kiosks = self.detect_inactive_kiosks()
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_kiosks': len(health_statuses),
            'active_kiosks': len([k for k in health_statuses if k['status'] == 'active']),
            'inactive_kiosks': len(inactive_kiosks),
            'health_statuses': health_statuses,
            'inactive_list': inactive_kiosks
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

    def send_kiosk_alert(self, kiosk_uuid, alert_type='inactivity'):
        """
        Enviar alerta para un kiosk
        
        Args:
            kiosk_uuid (str): UUID del kiosk
            alert_type (str): Tipo de alerta
        """
        health_status = self.check_kiosk_health(kiosk_uuid)
        
        # Ejemplo de envío de alerta (puede ser por email, Slack, etc.)
        alert_message = f"""
        Alerta de Kiosk: {alert_type.upper()}
        UUID: {kiosk_uuid}
        Nombre: {health_status.get('name', 'N/A')}
        Estado: {health_status.get('status', 'Desconocido')}
        Última conexión: {health_status.get('last_online', 'Nunca')}
        """
        
        # Aquí podrías implementar el envío de alertas por diferentes canales
        self.logger.warning(alert_message)

def main():
    """
    Función principal para ejecutar el monitoreo de kiosks
    """
    parser = argparse.ArgumentParser(description='Monitor de Kiosks')
    parser.add_argument('--action', choices=['health', 'inactive', 'report', 'alert'], 
                        default='report', help='Acción a realizar')
    parser.add_argument('--uuid', help='UUID del kiosk para acciones específicas')
    parser.add_argument('--output', help='Archivo de salida para el informe')
    parser.add_argument('--days', type=int, default=7, 
                        help='Días de inactividad para detectar kiosks inactivos')
    
    args = parser.parse_args()
    
    monitor = KioskMonitor()
    
    try:
        if args.action == 'health':
            if args.uuid:
                print(json.dumps(monitor.check_kiosk_health(args.uuid), indent=2))
            else:
                print(json.dumps(monitor.check_all_kiosks_health(), indent=2))
        
        elif args.action == 'inactive':
            inactive_kiosks = monitor.detect_inactive_kiosks(args.days)
            print(json.dumps(inactive_kiosks, indent=2))
        
        elif args.action == 'report':
            report = monitor.generate_kiosk_report(args.output)
            print(json.dumps(report, indent=2))
        
        elif args.action == 'alert':
            if not args.uuid:
                print("Error: Se requiere un UUID para enviar una alerta")
                sys.exit(1)
            
            monitor.send_kiosk_alert(args.uuid)
    
    except Exception as e:
        logging.error(f"Error en el monitoreo de kiosks: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 