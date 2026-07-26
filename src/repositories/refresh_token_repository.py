"""
Refresh token repository.

Refresh-token rotation needs persistence so a previously used token can be
revoked and rejected on replay. Callers store and look up only the hashed
JWT `jti`, never the bearer token itself.
"""
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.security import hash_token_identifier
from src.models.refresh_token_model import RefreshToken
from src.repositories.base_repository import BaseRepository


class RefreshTokenRepository(BaseRepository[RefreshToken]):
    def __init__(self, db: Session) -> None:
        super().__init__(model=RefreshToken, db=db)

    def add_for_user(self, user_id: UUID, token_id: UUID, expires_at: datetime) -> RefreshToken:
        refresh_token = RefreshToken(
            user_id=user_id,
            token_hash=hash_token_identifier(token_id),
            expires_at=expires_at,
        )
        self.db.add(refresh_token)
        self.db.flush()
        return refresh_token

    def get_active_by_token_id(self, token_id: str | UUID) -> RefreshToken | None:
        now = datetime.now(timezone.utc)
        stmt = select(RefreshToken).where(
            RefreshToken.token_hash == hash_token_identifier(token_id),
            RefreshToken.revoked_at.is_(None),
            RefreshToken.expires_at > now,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def revoke(
        self,
        refresh_token: RefreshToken,
        replaced_by_token_id: UUID | None = None,
    ) -> RefreshToken:
        refresh_token.revoked_at = datetime.now(timezone.utc)
        refresh_token.replaced_by_token_id = replaced_by_token_id
        self.db.flush()
        return refresh_token

    def revoke_all_for_user(self, user_id: UUID) -> int:
        tokens = self.db.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked_at.is_(None),
            )
        ).scalars()
        count = 0
        now = datetime.now(timezone.utc)
        for token in tokens:
            token.revoked_at = now
            count += 1
        self.db.flush()
        return count
