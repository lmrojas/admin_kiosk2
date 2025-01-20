# Tests del Sistema Admin Kiosk

## Estructura de Directorios

- `unit/`: Tests unitarios para modelos y servicios
- `integration/`: Tests de integración
- `e2e/`: Tests de extremo a extremo

## Requisitos

- Python 3.9+
- pytest
- pytest-cov

## Instalación de Dependencias

```bash
pip install -r requirements.txt
```

## Ejecución de Tests

### Ejecutar todos los tests

```bash
pytest
```

### Ejecutar tests con cobertura

```bash
pytest --cov=app
```

### Ejecutar tests específicos

```bash
# Tests unitarios
pytest tests/unit/

# Tests de un módulo específico
pytest tests/unit/test_user_model.py
```

## Configuración

La configuración de tests se encuentra en `pytest.ini`:
- Usa base de datos PostgreSQL para pruebas (admin_kiosk2_test)
- Genera reportes de cobertura
- Configura opciones de pytest

## Buenas Prácticas

- Cada test debe ser independiente
- Usar fixtures para configuración
- Cubrir casos de éxito y error
- Mantener tests pequeños y enfocados

## Contribución

- Agregar tests para nuevas funcionalidades
- Mantener cobertura de código cercana al 100%
- Documentar casos de prueba 