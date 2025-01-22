# Documentación de Arquitectura - Admin Kiosk
Este código solo puede ser modificado según @cura.md y project_custom_structure.txt

## 1. Visión General del Sistema

### 1.1 Propósito
El sistema Admin Kiosk proporciona una plataforma centralizada para la gestión y monitoreo de kiosks, integrando capacidades de IA para la detección de anomalías y optimización de operaciones.

### 1.2 Arquitectura General
- **Patrón MVC (Model-View-Controller) con Flask**
- **Servicios Independientes**
- **API RESTful**
- **WebSockets para Monitoreo en Tiempo Real**

## 2. Componentes Principales

### 2.1 Backend (Flask + SQLAlchemy)
- **Models**: Definición de datos usando SQLAlchemy ORM
- **Views**: Blueprints y rutas de Flask
- **Services**: Lógica de negocio compleja y operaciones
- **Tasks**: Procesos asíncronos y trabajos programados

### 2.2 Frontend (React + TypeScript)
- **Componentes**: UI modular y reutilizable
- **Hooks**: Gestión de estado y efectos
- **Services**: Comunicación con API
- **WebSocket**: Conexiones en tiempo real

### 2.3 Sistema de IA
- **Módulo de Predicción**: Detección de anomalías
- **Módulo de Entrenamiento**: Actualización de modelos
- **Sistema de Explicabilidad**: Interpretación de predicciones
- **Monitoreo de Drift**: Detección de cambios en datos

## 3. Flujos de Datos

### 3.1 Autenticación y Autorización
```mermaid
sequenceDiagram
    Cliente->>+API: Login Request
    API->>+Auth Service: Validar Credenciales
    Auth Service->>+2FA Service: Verificar 2FA
    2FA Service-->>-Auth Service: Token 2FA
    Auth Service-->>-API: Token JWT
    API-->>-Cliente: Respuesta Auth
```

### 3.2 Monitoreo de Kiosks
```mermaid
sequenceDiagram
    Kiosk->>+WebSocket: Conexión
    WebSocket->>+Monitor Service: Datos en tiempo real
    Monitor Service->>+IA Service: Análisis
    IA Service-->>-Monitor Service: Predicciones
    Monitor Service-->>-WebSocket: Alertas
    WebSocket-->>-Kiosk: Actualizaciones
```

## 4. Seguridad

### 4.1 Capas de Seguridad
- Autenticación de dos factores (2FA)
- JWT con rotación de tokens
- Rate limiting con Flask-Limiter
- Validación de entrada
- CORS configurado con Flask-CORS
- Headers de seguridad personalizados

### 4.2 Protección de Datos
- Encriptación en tránsito (TLS)
- Encriptación en reposo
- Sanitización de entrada
- Auditoría de accesos con SQLAlchemy

## 5. Escalabilidad y Rendimiento

### 5.1 Estrategias de Caché
- Redis para caché distribuido
- Flask-Caching para consultas frecuentes
- Caché de sesiones con Flask-Session

### 5.2 Optimizaciones
- Índices de base de datos SQLAlchemy
- Consultas optimizadas
- Compresión de respuestas Flask
- Lazy loading de relaciones SQLAlchemy

## 6. Monitoreo y Logging

### 6.1 Métricas Clave
- Tiempo de respuesta
- Uso de recursos
- Errores y excepciones
- Métricas de negocio

### 6.2 Sistema de Logs
- Logs estructurados con Flask-Logger
- Niveles de severidad
- Rotación de logs
- Agregación centralizada

## 7. Despliegue

### 7.1 Infraestructura
- Contenedores Docker
- Orquestación con ECS
- Base de datos PostgreSQL
- Redis para caché y colas

### 7.2 CI/CD
- GitHub Actions
- Tests automatizados con pytest
- Despliegue continuo
- Monitoreo de salud 