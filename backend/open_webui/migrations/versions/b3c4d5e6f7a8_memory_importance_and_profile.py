"""Add importance_score, access_count, last_accessed_at to memory; add memory_profile table

Revision ID: b3c4d5e6f7a8
Revises: a2b3c4d5e6f7
Create Date: 2025-04-12 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'b3c4d5e6f7a8'
down_revision: Union[str, None] = 'a2b3c4d5e6f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCOPE_WEIGHTS = {'personal': 1.0, 'work': 0.9, 'preference': 0.8, 'general': 0.5}


def upgrade() -> None:
    # Extend memory table
    op.add_column('memory', sa.Column('importance_score', sa.Float(), nullable=True, server_default='0.5'))
    op.add_column('memory', sa.Column('access_count', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('memory', sa.Column('last_accessed_at', sa.BigInteger(), nullable=True))

    # User profile table — one row per user, regenerated every N new facts
    op.create_table(
        'memory_profile',
        sa.Column('user_id', sa.String(), primary_key=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('fact_count_at_generation', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('updated_at', sa.BigInteger(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('memory_profile')
    op.drop_column('memory', 'last_accessed_at')
    op.drop_column('memory', 'access_count')
    op.drop_column('memory', 'importance_score')
