"""add_kiosk_location_history

Revision ID: add_kiosk_location_history
Revises: 1467e664b7c1
Create Date: 2025-01-22 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'add_kiosk_location_history'
down_revision = '1467e664b7c1'
branch_labels = None
depends_on = None

def upgrade():
    # Verificar si el enum ya existe
    conn = op.get_bind()
    inspector = inspect(conn)
    existing_enums = inspector.get_enums()
    
    # Solo crear el enum si no existe
    if not any(enum['name'] == 'location_type_enum' for enum in existing_enums):
        op.execute("CREATE TYPE location_type_enum AS ENUM ('assigned', 'reported')")
    
    # Crear la tabla usando el enum existente o recién creado
    op.create_table('kiosk_location_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('kiosk_id', sa.Integer(), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('location_type', postgresql.ENUM('assigned', 'reported', name='location_type_enum', create_type=False), nullable=False),
        sa.Column('previous_latitude', sa.Float(), nullable=True),
        sa.Column('previous_longitude', sa.Float(), nullable=True),
        sa.Column('change_reason', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['kiosk_id'], ['kiosks.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Crear índices para optimizar consultas
    op.create_index(op.f('ix_kiosk_location_history_kiosk_id'), 'kiosk_location_history', ['kiosk_id'], unique=False)
    op.create_index(op.f('ix_kiosk_location_history_timestamp'), 'kiosk_location_history', ['timestamp'], unique=False)
    op.create_index(op.f('ix_kiosk_location_history_location_type'), 'kiosk_location_history', ['location_type'], unique=False)

def downgrade():
    # Eliminar índices
    op.drop_index(op.f('ix_kiosk_location_history_location_type'), table_name='kiosk_location_history')
    op.drop_index(op.f('ix_kiosk_location_history_timestamp'), table_name='kiosk_location_history')
    op.drop_index(op.f('ix_kiosk_location_history_kiosk_id'), table_name='kiosk_location_history')
    
    # Eliminar tabla
    op.drop_table('kiosk_location_history')
    
    # Verificar si el enum existe antes de intentar eliminarlo
    conn = op.get_bind()
    inspector = inspect(conn)
    existing_enums = inspector.get_enums()
    
    if any(enum['name'] == 'location_type_enum' for enum in existing_enums):
        op.execute('DROP TYPE location_type_enum') 