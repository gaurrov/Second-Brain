"""
User repository.

All direct SQL/ORM querying for the `users` table happens here — the
service layer never issues a query directly against `models.User`.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.user_model import User
from src.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, db: Session):
        super().__init__(model=User, db=db)

    def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        return self.db.execute(stmt).scalar_one_or_none()

    def get_by_username(self, username: str) -> User | None:
        stmt = select(User).where(User.username == username)
        return self.db.execute(stmt).scalar_one_or_none()

    def exists_by_email_or_username(self, email: str, username: str) -> bool:
        stmt = select(User.id).where(
            (User.email == email) | (User.username == username)
        )
        return self.db.execute(stmt).first() is not None
