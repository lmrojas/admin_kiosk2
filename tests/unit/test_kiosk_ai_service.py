# EL CÓDIGO DE ESTE ARCHIVO PUEDE MODIFICARSE UNICAMENTE Y 
# SOLAMENTE SIGUIENDO LO ESTABLECIDO EN 'cura.md' Y 'project_custom_structure.txt'

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.kiosk_ai_service import KioskAIService, KioskAnomalyModel

@pytest.fixture
def ai_service():
    """Fixture para el servicio de IA"""
    return KioskAIService()

@pytest.fixture
def sample_metrics():
    """Fixture para métricas de ejemplo"""
    return {
        'cpu_usage': 75.5,
        'memory_usage': 80.2,
        'network_latency': 120.0
    }

def test_model_architecture():
    """Prueba la arquitectura del modelo"""
    model = KioskAnomalyModel()
    
    # Verificar capas del modelo
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert isinstance(model.layers[1], torch.nn.ReLU)
    assert isinstance(model.layers[2], torch.nn.Dropout)
    
    # Verificar dimensiones
    assert model.layers[0].in_features == 3
    assert model.layers[0].out_features == 10

def test_normalize_data(ai_service, sample_metrics):
    """Prueba la normalización de datos"""
    normalized = ai_service.normalize_data(sample_metrics)
    
    # Verificar que los valores estén normalizados
    assert all(isinstance(v, float) for v in normalized)
    assert all(-10 <= v <= 10 for v in normalized)

@patch('torch.load')
def test_load_model_success(mock_load, ai_service):
    """Prueba la carga exitosa del modelo"""
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    
    loaded_model = ai_service.load_model()
    assert loaded_model == mock_model
    mock_load.assert_called_once()

@patch('torch.load')
def test_load_model_fallback(mock_load, ai_service):
    """Prueba el fallback cuando falla la carga del modelo"""
    mock_load.side_effect = Exception("Error loading model")
    
    model = ai_service.load_model()
    assert isinstance(model, KioskAnomalyModel)

def test_predict_anomaly(ai_service, sample_metrics):
    """Prueba la predicción de anomalías"""
    probability = ai_service.predict_anomaly(sample_metrics)
    
    # Verificar que la probabilidad sea válida
    assert isinstance(probability, float)
    assert 0 <= probability <= 1

@pytest.mark.parametrize("metrics,expected", [
    (
        {'cpu_usage': 95.0, 'memory_usage': 90.0, 'network_latency': 300.0},
        True
    ),
    (
        {'cpu_usage': 50.0, 'memory_usage': 60.0, 'network_latency': 100.0},
        False
    )
])
def test_predict_anomaly_thresholds(ai_service, metrics, expected):
    """Prueba diferentes umbrales para detección de anomalías"""
    probability = ai_service.predict_anomaly(metrics)
    is_anomaly = probability > 0.5
    
    assert is_anomaly == expected

def test_generate_synthetic_data(ai_service):
    """Prueba la generación de datos sintéticos"""
    num_samples = 100
    data = ai_service.generate_synthetic_data(num_samples)
    
    # Verificar estructura de datos
    assert len(data) == num_samples
    assert all(isinstance(d, dict) for d in data)
    assert all('cpu_usage' in d for d in data)
    assert all('memory_usage' in d for d in data)
    assert all('network_latency' in d for d in data)
    
    # Verificar rangos de valores
    for d in data:
        assert 0 <= d['cpu_usage'] <= 100
        assert 0 <= d['memory_usage'] <= 100
        assert 0 <= d['network_latency'] <= 500

def test_invalid_metrics(ai_service):
    """Prueba el manejo de métricas inválidas"""
    invalid_metrics = {
        'cpu_usage': -10,
        'memory_usage': 150,
        'network_latency': 'invalid'
    }
    
    with pytest.raises(ValueError):
        ai_service.predict_anomaly(invalid_metrics)

def test_model_training(ai_service):
    """Prueba el entrenamiento del modelo"""
    num_samples = 10
    epochs = 2
    
    # Generar datos de entrenamiento
    data = ai_service.generate_synthetic_data(num_samples)
    X = torch.tensor([[d['cpu_usage'], d['memory_usage'], d['network_latency']] 
                     for d in data]).float()
    y = torch.tensor([1 if d['cpu_usage'] > 90 else 0 for d in data]).float()
    
    # Entrenar modelo
    model = KioskAnomalyModel()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    initial_loss = None
    final_loss = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
    
    # Verificar que la pérdida disminuya
    assert final_loss <= initial_loss 