from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from . import config

_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, connect_args=_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    amount: Mapped[float] = mapped_column(Float)
    hour: Mapped[int] = mapped_column(Integer)
    distance_from_home_km: Mapped[float] = mapped_column(Float)
    merchant_risk: Mapped[float] = mapped_column(Float)
    txn_count_24h: Mapped[int] = mapped_column(Integer)
    is_foreign: Mapped[int] = mapped_column(Integer)
    card_age_days: Mapped[int] = mapped_column(Integer)
    fraud_probability: Mapped[float] = mapped_column(Float)
    is_fraud: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(20))


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
