import numpy as np
import pandas as pd

from satisfaction_prediction.pipelines.second_sprint_pipeline.nodes import (
    basic_clean,
    evaluate,
    split_data,
    train_baseline,
)


def test_basic_clean_removes_nan_and_encodes():

    df = pd.DataFrame(
        {
            "Gender": ["Female", "Male"],
            "Customer Type": ["Loyal Customer", "disloyal Customer"],
            "Type of Travel": ["Business travel", "Personal Travel"],
            "Class": ["Eco", "Business"],
            "satisfaction": ["satisfied", "neutral or dissatisfied"],
            "Inflight wifi service": [0, 4],
            "Departure/Arrival time convenient": [0, 3],
            "Ease of Online booking": [3, 0],
            "Gate location": [1, 2],
            "Food and drink": [3, 2],
            "Online boarding": [2, 3],
            "Seat comfort": [2, 0],
            "Inflight entertainment": [1, 0],
            "On-board service": [3, 4],
            "Leg room service": [1, 3],
            "Baggage handling": [2, 3],
            "Checkin service": [3, 1],
            "Inflight service": [2, 4],
            "Cleanliness": [3, 2],
            "Age": [22, 42],
            "Flight Distance": [400, 1500],
            "Departure Delay in Minutes": [None, 4],
            "Arrival Delay in Minutes": [None, 0],
        }
    )
    cleaned = basic_clean(df)

    assert not cleaned.isnull().any().any()
    assert set(cleaned["Gender"].unique()).issubset({0, 1})


def test_split_data_shapes():
    df = pd.DataFrame({"a": range(10), "b": range(10, 20), "target": [0, 1] * 5})
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="target", test_size=0.2, random_state=42
    )
    xtrs = 8
    xtsts = 2
    ytrs = 8
    ytsts = 2

    assert X_train.shape[0] == xtrs
    assert X_test.shape[0] == xtsts
    assert y_train.shape[0] == ytrs
    assert y_test.shape[0] == ytsts


def test_train_baseline_and_evaluate():
    X_train = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 2, 3]})
    y_train = np.array([0, 1, 0, 1])
    model = train_baseline(X_train, y_train)
    assert hasattr(model, "predict")
    preds = model.predict(X_train)

    assert set(preds).issubset({0, 1})

    metrics = evaluate(model, X_train, y_train)

    assert "roc_auc" in metrics or "rmse" in metrics
