import json
import datetime as dt
from sqlalchemy import create_engine, text
from src.api.config import settings

engine = create_engine(settings.DATABASE_URL, future=True)

def save_prediction(payload: dict, prediction: float, model_version: str):
    with engine.begin() as conn:

        if engine.url.get_backend_name() == "sqlite":
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                payload TEXT,
                prediction REAL,
                model_version TEXT
            )
            """))
        else:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions(
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP,
                payload JSONB,
                prediction DOUBLE PRECISION,
                model_version TEXT
            )
            """))

        conn.execute(text("""
        INSERT INTO predictions(ts, payload, prediction, model_version)
        VALUES (:ts, :payload, :pred, :ver)
        """), {
            "ts": dt.datetime.utcnow().isoformat(),
            "payload": json.dumps(payload),
            "pred": float(prediction),
            "ver": model_version
        })
