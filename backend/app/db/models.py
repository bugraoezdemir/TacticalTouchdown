"""SQLAlchemy ORM models for persistent player and team records."""
from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TeamRecord(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    players: Mapped[list[PlayerRecord]] = relationship("PlayerRecord", back_populates="team")


class PlayerRecord(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    nationality: Mapped[str] = mapped_column(String(50), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)

    # Game qualities (0–100)
    strength: Mapped[float] = mapped_column(Float, default=50.0)
    speed: Mapped[float] = mapped_column(Float, default=50.0)
    intelligence: Mapped[float] = mapped_column(Float, default=50.0)
    passing: Mapped[float] = mapped_column(Float, default=50.0)
    goalkeeping: Mapped[float] = mapped_column(Float, default=40.0)
    shooting: Mapped[float] = mapped_column(Float, default=50.0)
    dribbling: Mapped[float] = mapped_column(Float, default=50.0)

    # Lifetime statistics
    total_goals: Mapped[int] = mapped_column(Integer, default=0)
    total_assists: Mapped[int] = mapped_column(Integer, default=0)
    total_tackles: Mapped[int] = mapped_column(Integer, default=0)
    games_played: Mapped[int] = mapped_column(Integer, default=0)

    team: Mapped[TeamRecord] = relationship("TeamRecord", back_populates="players")

    @property
    def goals_per_game(self) -> float:
        return self.total_goals / self.games_played if self.games_played else 0.0

    @property
    def assists_per_game(self) -> float:
        return self.total_assists / self.games_played if self.games_played else 0.0

    @property
    def tackles_per_game(self) -> float:
        return self.total_tackles / self.games_played if self.games_played else 0.0
