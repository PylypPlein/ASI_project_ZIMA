# src/nodes.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split as sk_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error

def load_raw(data_path: str) -> pd.DataFrame:
    """Load raw CSV and strip column names."""
    df = pd.read_csv(data_path, sep=';')
    df.columns = df.columns.str.strip()
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: encode categorical, fill missing values, remove outliers."""
    # Encoding
    gender_map = {'Female': 0, 'Male': 1}
    customer_map = {'Loyal Customer': 1, 'disloyal Customer': 0}
    travel_map = {'Business travel': 1, 'Personal Travel': 0}
    class_map = {'Business': 2, 'Eco Plus': 1, 'Eco': 0}
    satisfaction_map = {'satisfied': 1, 'neutral or dissatisfied': 0}

    df['Gender'] = df['Gender'].map(gender_map)
    df['Customer Type'] = df['Customer Type'].map(customer_map)
    df['Type of Travel'] = df['Type of Travel'].map(travel_map)
    df['Class'] = df['Class'].map(class_map)
    df['satisfaction'] = df['satisfaction'].map(satisfaction_map)

    # Fill missing ratings with column mean
    rating_columns = [
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
        'Inflight service', 'Cleanliness'
    ]
    for col in rating_columns:
        mean_val = df.loc[df[col] != 0, col].mean()
        df[col] = df[col].replace(0, mean_val)

    # Fill delays with 0
    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(0)
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)

    # Remove outliers
    numeric_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    threshold = 5
    df = df[(z_scores < threshold).all(axis=1)]

    return df


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Split dataset into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = sk_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_baseline(X_train, y_train, params=None):
    """Train a simple logistic regression baseline model."""
    if params is None:
        params = {"max_iter": 200}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
        return {"roc_auc": roc_auc}
    except Exception:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {"rmse": rmse}
