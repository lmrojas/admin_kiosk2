"""
Controlador del dashboard de IA para Admin Kiosk.
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt
"""

from django.views.generic import TemplateView
from django.http import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from app.services.ai_metrics import AIMetricsService
from app.models.ai import ModelMetrics, PredictionLog, DriftMetrics
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AIDashboardView(LoginRequiredMixin, TemplateView):
    """Vista del dashboard de IA."""
    
    template_name = 'ai/dashboard.html'
    metrics_service = AIMetricsService()
    
    def get_context_data(self, **kwargs):
        """Obtiene el contexto para el template."""
        context = super().get_context_data(**kwargs)
        
        try:
            # Obtener versiones disponibles del modelo
            context['model_versions'] = self.metrics_service.get_model_versions()
            
            # Obtener métricas iniciales
            latest_version = context['model_versions'][0] if context['model_versions'] else None
            if latest_version:
                context.update(self.metrics_service.get_initial_metrics(latest_version))
            
        except Exception as e:
            logger.error(f"Error al obtener contexto del dashboard: {str(e)}")
            context['error'] = 'Error al cargar los datos del dashboard'
        
        return context

class AIMetricsAPI(LoginRequiredMixin):
    """API para obtener métricas del modelo."""
    
    metrics_service = AIMetricsService()
    
    def get(self, request):
        """Maneja peticiones GET."""
        try:
            # Obtener y validar parámetros
            params = self._validate_and_parse_params(request.GET)
            if 'error' in params:
                return JsonResponse({'error': params['error']}, status=400)
            
            # Obtener métricas del período
            metrics_data = self.metrics_service.get_metrics_for_period(**params)
            if 'error' in metrics_data:
                return JsonResponse({'error': metrics_data['error']}, status=400)
            
            return JsonResponse(metrics_data)
            
        except ValueError as e:
            return JsonResponse({
                'error': f'Error en el formato de los parámetros: {str(e)}'
            }, status=400)
            
        except Exception as e:
            logger.error(f"Error al obtener métricas: {str(e)}")
            return JsonResponse({
                'error': 'Error interno al procesar la solicitud'
            }, status=500)
    
    def _validate_and_parse_params(self, params):
        """Valida y parsea los parámetros de la petición."""
        try:
            version = params.get('model_version')
            if not version:
                return {'error': 'Se requiere especificar la versión del modelo'}
            
            start_date = datetime.strptime(params.get('start_date'), '%Y-%m-%d')
            end_date = datetime.strptime(params.get('end_date'), '%Y-%m-%d') + timedelta(days=1)
            
            return {
                'version': version,
                'start_date': start_date,
                'end_date': end_date
            }
            
        except (TypeError, ValueError) as e:
            return {'error': f'Error en el formato de las fechas: {str(e)}'} 