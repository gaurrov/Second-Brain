"""add composite indexes for message history queries

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-12 00:00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Conversation history reads:
    #   WHERE conversation_id = ? AND user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?
    op.create_index(
        "ix_messages_conversation_user_created",
        "messages",
        ["conversation_id", "user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_messages_conversation_user_created", table_name="messages")
