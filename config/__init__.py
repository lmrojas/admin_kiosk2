"""
Inicialización de configuraciones.
Expone las diferentes configuraciones de la aplicación.
"""

from .default import Config, DevelopmentConfig, TestingConfig, ProductionConfig

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 