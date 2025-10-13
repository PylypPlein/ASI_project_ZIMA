import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("../../data/01_raw/sample.csv", sep=";")
df.columns = df.columns.str.strip()

gender_map = {"Female": 0, "Male": 1}
customer_map = {"Loyal Customer": 1, "disloyal Customer": 0}
travel_map = {"Business travel": 1, "Personal Travel": 0}
class_map = {"Business": 2, "Eco Plus": 1, "Eco": 0}
satisfaction_map = {"satisfied": 1, "neutral or dissatisfied": 0}

df["Gender"] = df["Gender"].map(gender_map)
df["Customer Type"] = df["Customer Type"].map(customer_map)
df["Type of Travel"] = df["Type of Travel"].map(travel_map)
df["Class"] = df["Class"].map(class_map)
df["satisfaction"] = df["satisfaction"].map(satisfaction_map)

rating_columns = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

for col in rating_columns:
    mean_val = df.loc[df[col] != 0, col].mean()
    df[col] = df[col].replace(0, mean_val)

df["Departure Delay in Minutes"] = df["Departure Delay in Minutes"].fillna(0)
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)


numeric_cols = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

z_scores = np.abs(stats.zscore(df[numeric_cols]))
threshold = 5
df = df[(z_scores < threshold).all(axis=1)]


df.to_csv("../../data/01_raw/sample_clean.csv", index=False)
