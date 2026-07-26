"""add composite index on documents(user_id, created_at)

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-26 00:00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Composite index for the primary list query pattern:
    #   WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?
    op.create_index(
        "ix_documents_user_id_created_at",
        "documents",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_documents_user_id_created_at", table_name="documents")
