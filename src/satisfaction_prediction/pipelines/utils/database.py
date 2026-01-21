import json
import logging
import os
from datetime import datetime

from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/predictions.db")

engine = create_engine(DATABASE_URL, future=True)


def init_db():
    try:
        with engine.begin() as conn:

            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    model_version TEXT NOT NULL
                )
            """
                )
            )

        log.info("Tabela predictions zainicjalizowana.")
    except Exception as e:
        log.error(f"Błąd inicjalizacji bazy danych: {e}")
        raise


def save_prediction(payload: dict, prediction: float | int, model_version: str) -> int:
    init_db()

    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO predictions (ts, payload, prediction, model_version)
                    VALUES (:ts, :payload, :prediction, :model_version)
                """
                ),
                {
                    "ts": datetime.utcnow().isoformat(),
                    "payload": json.dumps(payload),
                    "prediction": float(prediction),
                    "model_version": model_version,
                },
            )
            record_id = result.lastrowid
            log.info(f"Predykcja zapisana do bazy z ID: {record_id}")
            return record_id
    except Exception as e:
        log.error(f"Błąd zapisu predykcji do bazy: {e}")
        raise


def get_predictions(limit: int = 100) -> list:
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, ts, payload, prediction, model_version
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT :limit
                """
                ),
                {"limit": limit},
            ).fetchall()
            return [dict(row._mapping) for row in rows]
    except Exception as e:
        log.error(f"Błąd pobierania predykcji z bazy: {e}")
        return []
