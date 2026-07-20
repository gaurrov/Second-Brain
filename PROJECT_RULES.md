# SecondBrain AI Backend Rules

## Architecture

- Follow Clean Architecture.
- Keep business logic in services.
- Keep routes thin.
- Use repository pattern.
- Never access the database directly from routes.
- Use dependency injection.
- Use async where appropriate.

## Authentication

JWT authentication.

Every protected endpoint must obtain the authenticated user.

## Multi-tenancy

Every resource belongs to one user.

Every database query must filter by user_id.

Every Qdrant search must filter by user_id.

Never allow cross-user retrieval.

## Code Style

- Type hints everywhere.
- Pydantic v2.
- SQLAlchemy 2.0.
- No duplicated logic.
- Proper logging.
- Proper exception handling.