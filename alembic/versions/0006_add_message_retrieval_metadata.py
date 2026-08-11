"""add retrieval_metadata to messages

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-11 00:00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("retrieval_metadata", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("messages", "retrieval_metadata")
