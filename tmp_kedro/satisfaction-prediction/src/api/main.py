from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Features(BaseModel):
    id: int
    Gender: str
    Customer_Type: str
    Age: int
    Type_of_Travel: str
    Class: str
    Flight_Distance: int
    Inflight_wifi_service: int
    Departure_Arrival_time_convenient: int
    Ease_of_Online_booking: int
    Gate_location: int
    Food_and_drink: int
    Online_boarding: int
    Seat_comfort: int
    Inflight_entertainment: int
    On_board_service: int
    Leg_room_service: int
    Baggage_handling: int
    Checkin_service: int
    Inflight_service: int
    Cleanliness: int
    Departure_Delay_in_Minutes: int
    Arrival_Delay_in_Minutes: float | int


class Prediction(BaseModel):
    prediction: float | int
    model_version: str


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    return {"prediction": 0.0, "model_version": "local-dev"}
