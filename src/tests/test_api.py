from fastapi.testclient import TestClient
from sqlalchemy import text

from satisfaction_prediction.pipelines.utils.database import engine
from src.api.main import app


client = TestClient(app)


def test_validation_error():
    payload = {
        "lp": 1,
        "id": 1,
        "Gender": "Male",
        "Customer_Type": "Loyal Customer",
        "Age": 30,
        "Type_of_Travel": "Business travel",
        "Class": "Business",
        "Flight_Distance": "oops",
    }

    r = client.post("/predict", json=payload)

    assert r.status_code == 422  # noqa: PLR2004


def test_predict_integration_db():
    valid_payload = {
        "lp": 1001,
        "id": 1001,
        "Gender": "Male",
        "Customer Type": "Loyal Customer",
        "Age": 30,
        "Type of Travel": "Business travel",
        "Class": "Business",
        "Flight Distance": 200,
        "Inflight wifi service": 5,
        "Departure/Arrival time convenient": 5,
        "Ease of Online booking": 5,
        "Gate location": 5,
        "Food and drink": 5,
        "Online boarding": 5,
        "Seat comfort": 5,
        "Inflight entertainment": 5,
        "On-board service": 5,
        "Leg room service": 5,
        "Baggage handling": 5,
        "Checkin service": 5,
        "Inflight service": 5,
        "Cleanliness": 5,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0.0,
    }

    initial_count = 0
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM predictions"))
            initial_count = result.scalar()
    except Exception:
        initial_count = 0

    r = client.post("/predict", json=valid_payload)

    assert r.status_code == 200  # noqa: PLR2004
    assert "prediction" in r.json()

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM predictions"))
        final_count = result.scalar()

    assert final_count == initial_count + 1
