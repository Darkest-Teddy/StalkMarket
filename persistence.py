import copy
import os
from typing import Optional

try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        Float,
        String,
        ForeignKey,
        JSON,
        select,
        delete,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    _HAS_SQLA = True
except Exception:
    _HAS_SQLA = False
    create_engine = Column = Integer = Float = String = ForeignKey = JSON = select = delete = None  # type: ignore
    declarative_base = sessionmaker = None  # type: ignore

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
_ENGINE = None
_SessionLocal = None
_Base = None

if _HAS_SQLA and DATABASE_URL:
    _ENGINE = create_engine(DATABASE_URL)
    _SessionLocal = sessionmaker(bind=_ENGINE)
    _Base = declarative_base()

    class SeasonORM(_Base):
        __tablename__ = "seasons"
        season_id = Column(String, primary_key=True)
        meta = Column(JSON)

    class PriceORM(_Base):
        __tablename__ = "prices"
        id = Column(Integer, primary_key=True, autoincrement=True)
        season_id = Column(String, ForeignKey("seasons.season_id", ondelete="CASCADE"))
        ts = Column(Integer)
        crop_id = Column(String)
        price = Column(Float)

    class EventORM(_Base):
        __tablename__ = "events"
        id = Column(Integer, primary_key=True, autoincrement=True)
        season_id = Column(String, ForeignKey("seasons.season_id", ondelete="CASCADE"))
        ts = Column(Integer)
        payload = Column(JSON)

    class SeasonStatsORM(_Base):
        __tablename__ = "season_stats"
        season_id = Column(String, ForeignKey("seasons.season_id", ondelete="CASCADE"), primary_key=True)
        metrics = Column(JSON)
        money_generated = Column(Float)

    class UserStatsORM(_Base):
        __tablename__ = "user_stats"
        player_id = Column(String, primary_key=True)
        data = Column(JSON)
        money_generated = Column(Float)
else:
    SeasonORM = PriceORM = EventORM = SeasonStatsORM = UserStatsORM = None  # type: ignore


def persistence_enabled() -> bool:
    return _ENGINE is not None and _SessionLocal is not None and _Base is not None


def init_persistence(memory_store) -> None:
    if not persistence_enabled():
        if DATABASE_URL and not _HAS_SQLA:
            print("[WARN] SQLAlchemy not installed; persistence disabled")
        return
    _Base.metadata.create_all(bind=_ENGINE)
    load_memory_from_db(memory_store)


def load_memory_from_db(memory_store) -> None:
    if not persistence_enabled():
        return
    with _SessionLocal() as session:
        seasons = session.scalars(select(SeasonORM)).all()
        for season in seasons:
            meta = copy.deepcopy(season.meta or {})
            params = meta.pop("season_params", None)
            memory_store.seasons[season.season_id] = meta
            if params:
                memory_store.season_params[season.season_id] = params
            price_rows = session.scalars(
                select(PriceORM).where(PriceORM.season_id == season.season_id).order_by(PriceORM.ts)
            ).all()
            memory_store.prices[season.season_id] = [
                dict(ts=row.ts, crop_id=row.crop_id, price=row.price) for row in price_rows
            ]
            event_rows = session.scalars(
                select(EventORM).where(EventORM.season_id == season.season_id).order_by(EventORM.ts)
            ).all()
            memory_store.events[season.season_id] = [row.payload for row in event_rows]


def persist_season_state(memory_store, season_id: str) -> None:
    if not persistence_enabled():
        return
    prices = memory_store.prices.get(season_id, [])
    events = memory_store.events.get(season_id, [])
    meta = copy.deepcopy(memory_store.seasons.get(season_id, {}))
    meta["season_params"] = memory_store.season_params.get(season_id, [])
    with _SessionLocal() as session:
        session.merge(SeasonORM(season_id=season_id, meta=meta))
        session.execute(delete(PriceORM).where(PriceORM.season_id == season_id))
        session.execute(delete(EventORM).where(EventORM.season_id == season_id))
        if prices:
            session.bulk_save_objects(
                [
                    PriceORM(
                        season_id=season_id,
                        ts=int(rec["ts"]),
                        crop_id=rec["crop_id"],
                        price=float(rec["price"]),
                    )
                    for rec in prices
                ]
            )
        if events:
            session.bulk_save_objects(
                [
                    EventORM(
                        season_id=season_id,
                        ts=int(rec.get("ts", 0)),
                        payload=rec,
                    )
                    for rec in events
                ]
            )
        session.commit()


def persist_report_stats(season_id: str, metrics: dict, player_id: Optional[str] = None) -> None:
    if not persistence_enabled():
        return
    money_generated = float(metrics.get("wealth", 0.0) or 0.0)
    with _SessionLocal() as session:
        session.merge(SeasonStatsORM(season_id=season_id, metrics=metrics, money_generated=money_generated))
        if player_id:
            session.merge(UserStatsORM(player_id=player_id, data=metrics, money_generated=money_generated))
        session.commit()
