import requests, os, streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.title("ASI ML project")

feature_num = st.number_input("feature_num (Flight Distance)", value=200.0)
feature_cat = st.selectbox("feature_cat (A=Male, B=Female)", ["A", "B"])

gender_map = {
    "A": "Male",
    "B": "Female",
}

if st.button("Predict"):
    payload = {
        "lp": 1,
        "id": 1,
        "Gender": gender_map[feature_cat],
        "Customer Type": "Loyal Customer",
        "Age": 30,
        "Type of Travel": "Business travel",
        "Class": "Business",
        "Flight Distance": int(feature_num),
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

    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

    st.write("Status:", r.status_code)
    st.json(r.json())