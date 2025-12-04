from fastapi.testclient import TestClient

from src.api.db import engine
from src.api.main import app
from sqlalchemy import text


client = TestClient(app)

def test_validation_error():
    payload = {
        "id": 1,
        "Gender": "Male",
        "Customer_Type": "Loyal Customer",
        "Age": 30,
        "Type_of_Travel": "Business travel",
        "Class": "Business",
        "Flight_Distance": "oops"
    }

    r = client.post("/predict", json=payload)

    assert r.status_code == 422




def test_predict_integration_db():
    valid_payload = {
        "id": 1001,
        "Gender": "Male",
        "Customer_Type": "Loyal Customer",
        "Age": 30,
        "Type_of_Travel": "Business travel",
        "Class": "Business",
        "Flight_Distance": 200,
        "Inflight_wifi_service": 5,
        "Departure_Arrival_time_convenient": 5,
        "Ease_of_Online_booking": 5,
        "Gate_location": 5,
        "Food_and_drink": 5,
        "Online_boarding": 5,
        "Seat_comfort": 5,
        "Inflight_entertainment": 5,
        "On_board_service": 5,
        "Leg_room_service": 5,
        "Baggage_handling": 5,
        "Checkin_service": 5,
        "Inflight_service": 5,
        "Cleanliness": 5,
        "Departure_Delay_in_Minutes": 0,
        "Arrival_Delay_in_Minutes": 0.0
    }

    initial_count = 0
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM predictions"))
            initial_count = result.scalar()
    except Exception:
        initial_count = 0

    r = client.post("/predict", json=valid_payload)

    assert r.status_code == 200
    assert "prediction" in r.json()

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM predictions"))
        final_count = result.scalar()

    assert final_count == initial_count + 1