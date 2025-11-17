import pandas as pd
import numpy as np
from typing import Any, Dict
import logging

# --- Wymagane importy ---
from autogluon.tabular import TabularPredictor
from scipy import stats
from sklearn.model_selection import train_test_split as sk_split

# Ustawienie loggera
log = logging.getLogger(__name__)


# ==============================================================================
# 1. FUNKCJE PRZETWARZANIA DANYCH
# (Niezbędne do przygotowania X_train, y_train dla AutoGluon)
# ==============================================================================

def load_raw(data_path: str) -> pd.DataFrame:
    """Load raw CSV and strip column names."""
    df = pd.read_csv(data_path, sep=";")
    df.columns = df.columns.str.strip()
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: encode categorical, fill missing values, remove outliers."""
    # Encoding
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

    # Fill missing ratings with column mean
    rating_columns = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
    ]
    for col in rating_columns:
        mean_val = df.loc[df[col] != 0, col].mean()
        df[col] = df[col].replace(0, mean_val)

    # Fill delays with 0
    df["Departure Delay in Minutes"] = df["Departure Delay in Minutes"].fillna(0)
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)

    # Remove outliers
    numeric_cols = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    threshold = 5
    df = df[(z_scores < threshold).all(axis=1)]

    return df


def split_data(
        df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
):
    """Split dataset into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = sk_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ==============================================================================
# 2. FUNKCJE AUTOGLUON (IMPLEMENTACJA ZADANIA 2)
# ==============================================================================

def train_autogluon(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any]) -> TabularPredictor:
    """
    Trenuje model AutoGluon na danych treningowych, używając konfiguracji z parameters.yml.

    Args:
        X_train: Ramka danych cech treningowych.
        y_train: Seria danych zmiennej celu treningowej.
        params: Parametry konfiguracyjne Kedro (powinny zawierać sekcję 'autogluon').

    Returns:
        Wytrenowany predyktor AutoGluon.
    """
    ag_params = params["autogluon"]
    label_col = ag_params["label"]
    
    # AutoGluon wymaga, aby cechy i etykieta były połączone
    train_data = X_train.copy()
    train_data[label_col] = y_train

    time_limit = ag_params.get("time_limit")
    presets = ag_params.get("presets")
    problem_type = ag_params.get("problem_type")
    eval_metric = ag_params.get("eval_metric")
    
    # Używamy globalnego seeda dla powtarzalności
    seed = params.get("run_params", {}).get("random_state", 42)

    log.info(f"Rozpoczynam trenowanie AutoGluon (Seed: {seed}, Limit: {time_limit}s)")

    # Ścieżka do zapisu tymczasowych modeli AG (wybieramy folder, który Kedro nie śledzi)
    save_path = "data/06_models/autogluon_temp_output"

    predictor = TabularPredictor(
        label=label_col,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=save_path,
        verbosity=2 
    ).fit(
        train_data=train_data,
        time_limit=time_limit,
        presets=presets,
        seed=seed,
    )

    log.info("Trening AutoGluon zakończony.")
    return predictor


def evaluate_autogluon(ag_predictor: TabularPredictor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Ocenia wytrenowany predyktor AutoGluon na zbiorze testowym i zwraca metryki do MetricsDataSet.
    """
    log.info("Rozpoczynanie ewaluacji AutoGluon na zbiorze testowym.")
    
    # Łączymy dane testowe dla wbudowanej metody evaluate
    temp_test_data = X_test.copy()
    temp_test_data[ag_predictor.label] = y_test
    
    # Ocena AutoGluon
    ag_metrics_raw = ag_predictor.evaluate(temp_test_data, silent=True)
    
    # Przygotowanie metryk dla Kedro MetricsDataSet
    metrics = {
        f"ag_test_score_{ag_predictor.eval_metric}": ag_metrics_raw[ag_predictor.eval_metric],
        "ag_test_error_rate": ag_metrics_raw.get('error_rate', 0.0),
        "ag_roc_auc": ag_metrics_raw.get('roc_auc', 0.0) 
    }

    log.info(f"Metryki AutoGluon na zbiorze testowym: {metrics}")

    # Feature Importance (logowanie)
    try:
        feature_importance_df = ag_predictor.feature_importance(
            X_test, 
            y_test, 
            model=ag_predictor.get_best_model(),
            subsample_size=50000 
        )
        log.info("\nTop 5 Feature Importance:")
        log.info("\n" + feature_importance_df.head(5).to_markdown())

    except Exception as e:
        log.warning(f"Nie udało się obliczyć Feature Importance: {e}")
    
    return metrics


def save_best_model(ag_predictor: TabularPredictor) -> TabularPredictor:
    """
    Zwraca obiekt TabularPredictor. Ten nod jest opcjonalny; jego głównym celem 
    jest zapisanie modelu do 'ag_model' przez PickleDataSet.
    """
    log.info(f"Zapisywanie obiektu TabularPredictor jako 'ag_model'.")
    return ag_predictor