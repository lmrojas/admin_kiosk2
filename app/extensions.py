"""
Extensiones de Flask centralizadas.
Siguiendo las mejores prácticas de Flask para evitar importaciones circulares.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate() 