"""add socket_id to kiosk

Revision ID: beabae057aa4
Revises: add_kiosk_location_history
Create Date: 2025-01-22 20:03:21.103117

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'beabae057aa4'
down_revision = 'add_kiosk_location_history'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('kiosk_location_history', schema=None) as batch_op:
        batch_op.drop_index('ix_kiosk_location_history_kiosk_id')
        batch_op.drop_index('ix_kiosk_location_history_location_type')
        batch_op.drop_index('ix_kiosk_location_history_timestamp')

    op.drop_table('kiosk_location_history')
    with op.batch_alter_table('kiosks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('socket_id', sa.String(length=50), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('kiosks', schema=None) as batch_op:
        batch_op.drop_column('socket_id')

    op.create_table('kiosk_location_history',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('kiosk_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('latitude', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('longitude', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('accuracy', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('timestamp', postgresql.TIMESTAMP(), autoincrement=False, nullable=False),
    sa.Column('location_type', postgresql.ENUM('assigned', 'reported', name='location_type_enum'), autoincrement=False, nullable=False),
    sa.Column('previous_latitude', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('previous_longitude', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('change_reason', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(), server_default=sa.text('now()'), autoincrement=False, nullable=False),
    sa.Column('created_by', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name='kiosk_location_history_created_by_fkey'),
    sa.ForeignKeyConstraint(['kiosk_id'], ['kiosks.id'], name='kiosk_location_history_kiosk_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='kiosk_location_history_pkey')
    )
    with op.batch_alter_table('kiosk_location_history', schema=None) as batch_op:
        batch_op.create_index('ix_kiosk_location_history_timestamp', ['timestamp'], unique=False)
        batch_op.create_index('ix_kiosk_location_history_location_type', ['location_type'], unique=False)
        batch_op.create_index('ix_kiosk_location_history_kiosk_id', ['kiosk_id'], unique=False)

    # ### end Alembic commands ###
