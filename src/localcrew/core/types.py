"""Custom SQLAlchemy types for database compatibility."""

from datetime import datetime, timezone

from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator, TypeEngine


class JSONType(TypeDecorator):
    """JSON type that uses JSONB on PostgreSQL and JSON on other databases.

    This allows tests to run on SQLite while production uses PostgreSQL's
    optimized JSONB type.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: TypeEngine) -> TypeEngine:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


def utcnow() -> datetime:
    """Return current UTC datetime.

    Uses timezone-aware datetime instead of deprecated datetime.utcnow().
    """
    return datetime.now(timezone.utc)
