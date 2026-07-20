"""
Alembic runtime configuration.

This wires Alembic to:
  1. the application's own Settings (so the DB URL always comes from
     src.core.config, never duplicated in alembic.ini), and
  2. the shared `Base.metadata` (via src.db.base_class) so that
     `alembic revision --autogenerate` can detect model changes.
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project root importable when Alembic runs as a standalone CLI.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.config import settings  # noqa: E402
from src.db.base_class import Base  # noqa: E402

config = context.config

# Inject the real database URL from application settings, overriding the
# placeholder in alembic.ini.
config.set_main_option("sqlalchemy.url", settings.sqlalchemy_database_uri)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL without a live DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
