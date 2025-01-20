"""
Modelos de seguridad para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from django.db import models
from django.contrib.postgres.fields import JSONField
from django.utils.translation import gettext_lazy as _

class SecurityAudit(models.Model):
    """Modelo para auditar accesos al sistema."""

    class EventType(models.TextChoices):
        ACCESS = 'ACCESS', _('Acceso')
        LOGIN = 'LOGIN', _('Inicio de sesión')
        LOGIN_FAILED = 'LOGIN_FAILED', _('Inicio de sesión fallido')
        LOGOUT = 'LOGOUT', _('Cierre de sesión')
        PASSWORD_CHANGE = 'PASSWORD_CHANGE', _('Cambio de contraseña')
        PASSWORD_RESET = 'PASSWORD_RESET', _('Restablecimiento de contraseña')
        TWO_FACTOR = 'TWO_FACTOR', _('Autenticación de dos factores')
        TOKEN_REFRESH = 'TOKEN_REFRESH', _('Actualización de token')

    ip_address = models.GenericIPAddressField(
        verbose_name=_('Dirección IP'),
        help_text=_('Dirección IP desde donde se realizó el acceso')
    )
    
    user_id = models.IntegerField(
        verbose_name=_('ID de Usuario'),
        null=True,
        blank=True,
        help_text=_('ID del usuario que realizó la acción')
    )
    
    timestamp = models.DateTimeField(
        verbose_name=_('Fecha y hora'),
        auto_now_add=True,
        help_text=_('Fecha y hora del evento')
    )
    
    event_type = models.CharField(
        verbose_name=_('Tipo de evento'),
        max_length=20,
        choices=EventType.choices,
        help_text=_('Tipo de evento de seguridad')
    )
    
    method = models.CharField(
        verbose_name=_('Método HTTP'),
        max_length=10,
        help_text=_('Método HTTP utilizado')
    )
    
    path = models.CharField(
        verbose_name=_('Ruta'),
        max_length=255,
        help_text=_('Ruta accedida')
    )
    
    user_agent = models.TextField(
        verbose_name=_('User Agent'),
        null=True,
        blank=True,
        help_text=_('User Agent del cliente')
    )
    
    metadata = JSONField(
        verbose_name=_('Metadata'),
        default=dict,
        help_text=_('Información adicional del evento')
    )

    class Meta:
        verbose_name = _('Auditoría de Seguridad')
        verbose_name_plural = _('Auditorías de Seguridad')
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['ip_address']),
            models.Index(fields=['event_type']),
            models.Index(fields=['user_id']),
        ]
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.get_event_type_display()} - {self.ip_address} - {self.timestamp}"

class SecurityEvent(models.Model):
    """Modelo para eventos de seguridad específicos."""

    class Severity(models.TextChoices):
        LOW = 'LOW', _('Baja')
        MEDIUM = 'MEDIUM', _('Media')
        HIGH = 'HIGH', _('Alta')
        CRITICAL = 'CRITICAL', _('Crítica')

    event_type = models.CharField(
        verbose_name=_('Tipo de evento'),
        max_length=50,
        help_text=_('Identificador del tipo de evento de seguridad')
    )
    
    description = models.TextField(
        verbose_name=_('Descripción'),
        help_text=_('Descripción detallada del evento')
    )
    
    severity = models.CharField(
        verbose_name=_('Severidad'),
        max_length=10,
        choices=Severity.choices,
        help_text=_('Nivel de severidad del evento')
    )
    
    timestamp = models.DateTimeField(
        verbose_name=_('Fecha y hora'),
        auto_now_add=True,
        help_text=_('Fecha y hora del evento')
    )
    
    user_id = models.IntegerField(
        verbose_name=_('ID de Usuario'),
        null=True,
        blank=True,
        help_text=_('ID del usuario relacionado con el evento')
    )
    
    metadata = JSONField(
        verbose_name=_('Metadata'),
        default=dict,
        help_text=_('Información adicional del evento')
    )
    
    resolved = models.BooleanField(
        verbose_name=_('Resuelto'),
        default=False,
        help_text=_('Indica si el evento ha sido resuelto')
    )
    
    resolution_notes = models.TextField(
        verbose_name=_('Notas de resolución'),
        null=True,
        blank=True,
        help_text=_('Notas sobre la resolución del evento')
    )
    
    resolved_by = models.IntegerField(
        verbose_name=_('Resuelto por'),
        null=True,
        blank=True,
        help_text=_('ID del usuario que resolvió el evento')
    )
    
    resolved_at = models.DateTimeField(
        verbose_name=_('Fecha de resolución'),
        null=True,
        blank=True,
        help_text=_('Fecha y hora de resolución del evento')
    )

    class Meta:
        verbose_name = _('Evento de Seguridad')
        verbose_name_plural = _('Eventos de Seguridad')
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['severity']),
            models.Index(fields=['event_type']),
            models.Index(fields=['resolved']),
        ]
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.event_type} - {self.severity} - {self.timestamp}"

    def resolve(self, user_id: int, notes: str) -> None:
        """Marca el evento como resuelto."""
        from django.utils import timezone
        
        self.resolved = True
        self.resolved_by = user_id
        self.resolution_notes = notes
        self.resolved_at = timezone.now()
        self.save() 