"""
This module exists solely so that Alembic's `--autogenerate` can discover
every ORM model. Alembic's env.py imports `Base` from here (via this
module) which, by the time this file has been imported, has had every
model class registered against it.

IMPORTANT: whenever a new model file is added under src/models/, import
it here as well, or Alembic will silently ignore it during autogenerate.
"""
from src.models.base import Base  # noqa: F401
from src.models.user_model import User  # noqa: F401
from src.models.document_model import Document  # noqa: F401
from src.models.refresh_token_model import RefreshToken  # noqa: F401
from src.models.conversation_model import Conversation  # noqa: F401
from src.models.message_model import Message  # noqa: F401

# Future models get imported here, e.g.:
# from src.models.rating_model import Rating  # noqa: F401
