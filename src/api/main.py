from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.api.database import save_prediction
import joblib
import pandas as pd
from src.api.model import predictor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://ui-xxxxx.run.app",  # URL Twojego UI w Cloud Run
    "http://localhost:8501"      # jeśli testujesz lokalne UI
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

model = joblib.load(
    "/app/data/06_models/ag_production.pkl"
)
model_version = "ag_production"


class Features(BaseModel):
    lp: int
    id: int
    Gender: str
    Customer_Type: str = Field(alias="Customer Type")
    Age: int
    Type_of_Travel: str = Field(alias="Type of Travel")
    Class: str
    Flight_Distance: int = Field(alias="Flight Distance")
    Inflight_wifi_service: int = Field(alias="Inflight wifi service")
    Departure_Arrival_time_convenient: int = Field(
        alias="Departure/Arrival time convenient"
    )
    Ease_of_Online_booking: int = Field(alias="Ease of Online booking")
    Gate_location: int = Field(alias="Gate location")
    Food_and_drink: int = Field(alias="Food and drink")
    Online_boarding: int = Field(alias="Online boarding")
    Seat_comfort: int = Field(alias="Seat comfort")
    Inflight_entertainment: int = Field(alias="Inflight entertainment")
    On_board_service: int = Field(alias="On-board service")
    Leg_room_service: int = Field(alias="Leg room service")
    Baggage_handling: int = Field(alias="Baggage handling")
    Checkin_service: int = Field(alias="Checkin service")
    Inflight_service: int = Field(alias="Inflight service")
    Cleanliness: int
    Departure_Delay_in_Minutes: int = Field(alias="Departure Delay in Minutes")
    Arrival_Delay_in_Minutes: float = Field(alias="Arrival Delay in Minutes")

    class ConfigDict:
        allow_population_by_field_name = True


class Prediction(BaseModel):
    prediction: float | int
    model_version: str


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


COLUMN_MAP = {
    "Customer_Type": "Customer Type",
    "Type_of_Travel": "Type of Travel",
    "Flight_Distance": "Flight Distance",
    "Inflight_wifi_service": "Inflight wifi service",
    "Departure_Arrival_time_convenient": "Departure/Arrival time convenient",
    "Ease_of_Online_booking": "Ease of Online booking",
    "Gate_location": "Gate location",
    "Food_and_drink": "Food and drink",
    "Online_boarding": "Online boarding",
    "Seat_comfort": "Seat comfort",
    "Inflight_entertainment": "Inflight entertainment",
    "On_board_service": "On-board service",
    "Leg_room_service": "Leg room service",
    "Baggage_handling": "Baggage handling",
    "Checkin_service": "Checkin service",
    "Inflight_service": "Inflight service",
    "Departure_Delay_in_Minutes": "Departure Delay in Minutes",
    "Arrival_Delay_in_Minutes": "Arrival Delay in Minutes",
}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    df = pd.DataFrame([payload.model_dump(by_alias=True)])
    df = df.rename(columns=COLUMN_MAP)

    class_mapping = {"Business": 2, "Eco Plus": 1, "Eco": 0}
    if "Class" in df.columns:
        df["Class"] = df["Class"].map(class_mapping)

    required_cols = model.feature_metadata_in.get_features()
    type_map = model.feature_metadata_in.type_group_map_special

    df = df[[col for col in required_cols if col in df.columns]]

    for col in ["feat1", "feat2"]:
        if col not in df.columns:
            df[col] = 0

    for col, dtype in type_map.items():
        if col in df.columns:
            if dtype is int:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype is float:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "category":
                df[col] = df[col].astype("category")

    df = df[model.feature_metadata_in.get_features()]

    pred = predictor.predict(df).iloc[0]

    save_prediction(payload.model_dump(), pred, model_version)

    return {"prediction": float(pred), "model_version": model_version}
