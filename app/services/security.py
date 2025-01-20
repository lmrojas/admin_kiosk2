"""
Servicio de auditoría de seguridad para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime
from django.db import models
from app.models.security import SecurityAudit, SecurityEvent
from config.security.hardening import SECURITY_LOGGING

logger = logging.getLogger('security')

class SecurityAuditService:
    """Servicio para gestionar la auditoría de seguridad."""

    def log_access(self, ip: str, method: str, path: str, user_id: Optional[int] = None) -> None:
        """Registra un intento de acceso al sistema."""
        try:
            SecurityAudit.objects.create(
                ip_address=ip,
                method=method,
                path=path,
                user_id=user_id,
                timestamp=datetime.now(),
                event_type='ACCESS'
            )
        except Exception as e:
            logger.error(f"Error al registrar acceso: {str(e)}", 
                        extra={'ip': ip})

    def log_security_event(self, 
                          event_type: str, 
                          description: str, 
                          severity: str,
                          metadata: Optional[Dict] = None,
                          user_id: Optional[int] = None) -> None:
        """Registra un evento de seguridad."""
        try:
            SecurityEvent.objects.create(
                event_type=event_type,
                description=description,
                severity=severity,
                metadata=metadata or {},
                user_id=user_id,
                timestamp=datetime.now()
            )
            
            if severity in ['HIGH', 'CRITICAL']:
                self._notify_security_team(
                    event_type=event_type,
                    description=description,
                    severity=severity,
                    metadata=metadata
                )
        except Exception as e:
            logger.error(f"Error al registrar evento de seguridad: {str(e)}")

    def get_security_events(self, 
                          start_date: datetime,
                          end_date: datetime,
                          severity: Optional[str] = None,
                          event_type: Optional[str] = None) -> models.QuerySet:
        """Obtiene eventos de seguridad filtrados."""
        query = SecurityEvent.objects.filter(
            timestamp__range=(start_date, end_date)
        )
        
        if severity:
            query = query.filter(severity=severity)
        
        if event_type:
            query = query.filter(event_type=event_type)
        
        return query.order_by('-timestamp')

    def get_access_history(self, 
                          ip: Optional[str] = None,
                          user_id: Optional[int] = None,
                          limit: int = 100) -> models.QuerySet:
        """Obtiene el historial de accesos."""
        query = SecurityAudit.objects.all()
        
        if ip:
            query = query.filter(ip_address=ip)
        
        if user_id:
            query = query.filter(user_id=user_id)
        
        return query.order_by('-timestamp')[:limit]

    def analyze_security_patterns(self) -> List[Dict]:
        """Analiza patrones de seguridad para detectar anomalías."""
        patterns = []
        
        # Detectar intentos de acceso fallidos múltiples
        failed_attempts = self._analyze_failed_attempts()
        if failed_attempts:
            patterns.extend(failed_attempts)
        
        # Detectar accesos desde IPs sospechosas
        suspicious_ips = self._analyze_suspicious_ips()
        if suspicious_ips:
            patterns.extend(suspicious_ips)
        
        # Detectar patrones de acceso inusuales
        unusual_patterns = self._analyze_unusual_patterns()
        if unusual_patterns:
            patterns.extend(unusual_patterns)
        
        return patterns

    def _notify_security_team(self, 
                            event_type: str,
                            description: str,
                            severity: str,
                            metadata: Optional[Dict] = None) -> None:
        """Notifica al equipo de seguridad sobre eventos críticos."""
        from app.services.notification import NotificationService
        
        notification_service = NotificationService()
        message = f"""
        ¡Alerta de Seguridad!
        Tipo: {event_type}
        Severidad: {severity}
        Descripción: {description}
        Metadata: {metadata or {}}
        Timestamp: {datetime.now()}
        """
        
        notification_service.send_security_alert(message)

    def _analyze_failed_attempts(self) -> List[Dict]:
        """Analiza intentos de acceso fallidos para detectar patrones sospechosos."""
        threshold = 5  # Número de intentos fallidos antes de considerar sospechoso
        window = 300  # Ventana de tiempo en segundos (5 minutos)
        
        patterns = []
        failed_attempts = SecurityAudit.objects.filter(
            event_type='LOGIN_FAILED',
            timestamp__gte=datetime.now().timestamp() - window
        ).values('ip_address').annotate(
            count=models.Count('id')
        ).filter(count__gte=threshold)
        
        for attempt in failed_attempts:
            patterns.append({
                'type': 'MULTIPLE_FAILED_ATTEMPTS',
                'ip': attempt['ip_address'],
                'count': attempt['count'],
                'window': window,
                'severity': 'HIGH'
            })
        
        return patterns

    def _analyze_suspicious_ips(self) -> List[Dict]:
        """Analiza IPs para detectar comportamiento sospechoso."""
        from app.services.ip_intelligence import IPIntelligenceService
        
        patterns = []
        ip_service = IPIntelligenceService()
        
        recent_ips = SecurityAudit.objects.filter(
            timestamp__gte=datetime.now().timestamp() - 3600
        ).values_list('ip_address', flat=True).distinct()
        
        for ip in recent_ips:
            if ip_service.is_suspicious(ip):
                patterns.append({
                    'type': 'SUSPICIOUS_IP',
                    'ip': ip,
                    'reason': ip_service.get_threat_info(ip),
                    'severity': 'HIGH'
                })
        
        return patterns

    def _analyze_unusual_patterns(self) -> List[Dict]:
        """Analiza patrones de acceso inusuales."""
        patterns = []
        
        # Detectar accesos fuera de horario normal
        unusual_hours = self._detect_unusual_hours()
        if unusual_hours:
            patterns.extend(unusual_hours)
        
        # Detectar accesos desde ubicaciones inusuales
        unusual_locations = self._detect_unusual_locations()
        if unusual_locations:
            patterns.extend(unusual_locations)
        
        return patterns

    def _detect_unusual_hours(self) -> List[Dict]:
        """Detecta accesos fuera del horario laboral normal."""
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Fuera de 6 AM - 10 PM
            recent_accesses = SecurityAudit.objects.filter(
                timestamp__hour=current_hour
            ).values('ip_address', 'user_id').distinct()
            
            return [{
                'type': 'UNUSUAL_ACCESS_HOUR',
                'hour': current_hour,
                'ip': access['ip_address'],
                'user_id': access['user_id'],
                'severity': 'MEDIUM'
            } for access in recent_accesses]
        
        return []

    def _detect_unusual_locations(self) -> List[Dict]:
        """Detecta accesos desde ubicaciones geográficas inusuales."""
        from app.services.geolocation import GeolocationService
        
        patterns = []
        geo_service = GeolocationService()
        
        recent_accesses = SecurityAudit.objects.filter(
            timestamp__gte=datetime.now().timestamp() - 3600
        ).values('ip_address', 'user_id').distinct()
        
        for access in recent_accesses:
            location = geo_service.get_location(access['ip_address'])
            if geo_service.is_unusual_location(location, access['user_id']):
                patterns.append({
                    'type': 'UNUSUAL_LOCATION',
                    'ip': access['ip_address'],
                    'location': location,
                    'user_id': access['user_id'],
                    'severity': 'HIGH'
                })
        
        return patterns 