"""Extend memory table with scope and source_date

Revision ID: a2b3c4d5e6f7
Revises: b2c3d4e5f6a7
Create Date: 2025-04-12 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'a2b3c4d5e6f7'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # scope: 'personal' | 'work' | 'preference' | 'general'
    op.add_column('memory', sa.Column('scope', sa.String(), nullable=True, server_default='general'))
    # source_date: epoch timestamp of the conversation turn this fact came from
    op.add_column('memory', sa.Column('source_date', sa.BigInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column('memory', 'source_date')
    op.drop_column('memory', 'scope')
