"""Add formulations table for saved optimisation and simulation runs

Revision ID: 7b2e4c1d9a10
Revises: 65d831bd92a6
Create Date: 2026-11-20 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = '7b2e4c1d9a10'
down_revision = '65d831bd92a6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'formulations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('name', sa.String(150), nullable=False),
        sa.Column('kind', sa.String(20), nullable=False),
        sa.Column('request_json', sa.Text(), nullable=False),
        sa.Column('result_json', sa.Text(), nullable=False),
        sa.Column('feasible', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('protein', sa.Float(), nullable=True),
        sa.Column('cost_per_kg', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_formulations_user_id', 'formulations', ['user_id'])


def downgrade():
    op.drop_index('ix_formulations_user_id', table_name='formulations')
    op.drop_table('formulations')
