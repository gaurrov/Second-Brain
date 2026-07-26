"""
Generic base repository.

Provides common CRUD operations parameterized over an ORM model type.
Domain-specific repositories (e.g. UserRepository) inherit from this and
add query methods specific to that aggregate.
"""
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")


class BaseRepository(Generic[ModelType]):
    def __init__(self, model: type[ModelType], db: Session) -> None:
        self.model = model
        self.db = db

    def get_by_id(self, record_id: UUID) -> ModelType | None:
        return self.db.get(self.model, record_id)

    def create(self, obj: ModelType) -> ModelType:
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def delete(self, obj: ModelType) -> None:
        self.db.delete(obj)
        self.db.commit()
