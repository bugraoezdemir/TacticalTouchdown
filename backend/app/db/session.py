"""Async SQLAlchemy session factory (SQLite for development)."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = "sqlite+aiosqlite:///./football.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncSession:  # type: ignore[return]
    async with AsyncSessionLocal() as session:
        yield session
