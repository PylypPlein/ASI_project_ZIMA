import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from scipy import stats
from sklearn.model_selection import train_test_split

import wandb
from satisfaction_prediction.pipelines.utils.database import save_prediction

# Ustawienie loggera
log = logging.getLogger(__name__)

# --- Konfiguracja W&B ---
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "asi-ml-satisfaction")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
wandb_run = None


# ==============================================================================
# 1. FUNKCJE PRZETWARZANIA DANYCH
# ==============================================================================


def load_raw(data_path: str) -> pd.DataFrame:
    """Load raw CSV and strip column names."""
    df = pd.read_csv(data_path, sep=";")
    df.columns = df.columns.str.strip()
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: encode categorical, fill missing values, remove outliers."""

    df.columns = df.columns.str.strip()

    mappings = {
        "Gender": {"Female": 0, "Male": 1},
        "Customer Type": {"Loyal Customer": 1, "disloyal Customer": 0},
        "Type of Travel": {"Business travel": 1, "Personal Travel": 0},
        "Class": {"Business": 2, "Eco Plus": 1, "Eco": 0},
        "satisfaction": {"satisfied": 1, "neutral or dissatisfied": 0},
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

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
        if col in df.columns:
            mean_val = df.loc[df[col] != 0, col].mean()
            df[col] = df[col].replace(0, mean_val)

    for col in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    numeric_cols = [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        threshold = 5
        df = df[(z_scores < threshold).all(axis=1)]

    return df


def split_data(df: pd.DataFrame, target_col: str, run_params: dict):
    """Split dataset into train/test sets with optional stratification."""

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    split_params = run_params.get("split", {})
    test_size = split_params.get("test_size", 0.2)
    random_state = split_params.get("random_state", 42)
    stratify_flag = split_params.get("stratify", False)
    stratify = y if stratify_flag else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    y_train = y_train.to_frame() if isinstance(y_train, pd.Series) else y_train
    y_test = y_test.to_frame() if isinstance(y_test, pd.Series) else y_test

    return X_train, X_test, y_train, y_test


# ==============================================================================
# 2. FUNKCJE AUTOGLUON + W&B
# ==============================================================================
MIN_TRAIN_ROWS = 2


def train_autogluon(X_train, y_train, params):
    """
    Trenuje model AutoGluon TabularPredictor.
    Każdy `kedro run` = jeden eksperyment zalogowany w W&B.
    """

    global wandb_run

    label_col = params["label"]

    train_data = X_train.copy()
    train_data[label_col] = y_train

    mask = train_data[label_col].notna() & np.isfinite(train_data[label_col])
    train_data = train_data.loc[mask].reset_index(drop=True)

    if train_data.shape[0] < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Za mało wierszy do trenowania po oczyszczeniu danych: {train_data.shape[0]}. "
            "Sprawdź dane źródłowe i kolumnę etykiet."
        )

    time_limit = params.get("time_limit")
    presets = params.get("presets")
    problem_type = params.get("problem_type")
    eval_metric = params.get("eval_metric")
    seed = params.get("random_state", 42)

    save_path = "data/06_models/autogluon_temp_output"

    # --- Start runu W&B i zapis parametrów eksperymentu ---
    wandb_config = {
        "autogluon": {
            "time_limit": time_limit,
            "presets": presets,
            "problem_type": problem_type,
            "eval_metric": eval_metric,
        },
        "label": label_col,
        "n_train_rows": len(train_data),
        "n_features": len(X_train.columns),
        "random_state": seed,
    }

    try:
        log.info(
            f"Uruchamianie eksperymentu W&B: project={WANDB_PROJECT}, entity={WANDB_ENTITY}"
        )
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="ag-train",
            config=wandb_config,
        )
    except Exception as e:
        log.warning(
            f"Nie udało się zainicjalizować W&B (działam dalej bez logowania): {e}"
        )
        wandb_run = None

    # --- Trening AutoGluon + logowanie czasu treningu do W&B ---
    start_time = time.time()
    predictor = TabularPredictor(
        label=label_col,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=save_path,
        verbosity=2,
    ).fit(
        train_data=train_data,
        time_limit=time_limit,
        presets=presets,
        ag_args_fit={"seed": seed},
    )
    train_time_s = time.time() - start_time

    if wandb_run is not None:
        try:
            wandb_run.log({"train_time_s": train_time_s})
        except Exception as e:
            log.warning(f"Nie udało się zalogować czasu treningu do W&B: {e}")

    return predictor


def evaluate_autogluon(
    ag_predictor: TabularPredictor, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, any]:
    """
    Ocenia wytrenowany predyktor AutoGluon na zbiorze testowym i zwraca metryki do MetricsDataSet.
    Dodatkowo loguje metryki i feature importance do W&B.
    """
    global wandb_run

    log.info("Rozpoczynanie ewaluacji AutoGluon na zbiorze testowym.")

    temp_test_data = X_test.copy()
    temp_test_data[ag_predictor.label] = y_test

    ag_metrics_raw = ag_predictor.evaluate(temp_test_data, silent=True)

    metrics = {
        f"ag_test_score_{ag_predictor.eval_metric}": ag_metrics_raw.get(
            ag_predictor.eval_metric, 0.0
        ),
        "ag_test_error_rate": ag_metrics_raw.get("error_rate", 0.0),
        "ag_roc_auc": ag_metrics_raw.get("roc_auc", None),
    }

    log.info(f"Metryki AutoGluon na zbiorze testowym: {metrics}")

    # --- Logowanie metryk testowych do W&B ---
    if wandb_run is not None:
        try:
            log_dict = {k: float(v) for k, v in metrics.items() if v is not None}
            wandb_run.log(log_dict)
        except Exception as e:
            log.warning(f"Nie udało się zalogować metryk testowych do W&B: {e}")

    # --- Feature importance + logowanie top 5 do W&B ---
    try:
        feature_importance_df = ag_predictor.feature_importance(
            temp_test_data, subsample_size=50000
        )
        log.info("\nTop 5 Feature Importance:")
        log.info("\n" + feature_importance_df.head(5).to_markdown())

        if wandb_run is not None:
            wandb_run.log(
                {
                    "feature_importance_top5": wandb.Table(
                        dataframe=feature_importance_df.head(5)
                    )
                }
            )

    except Exception as e:
        log.warning(f"Nie udało się obliczyć Feature Importance: {e}")

    return metrics


def save_best_model(ag_predictor: TabularPredictor) -> TabularPredictor:
    """
    Zwraca obiekt TabularPredictor.
    Ten node:
    - pozwala Kedro zapisać model do 'ag_model' (PickleDataSet),
    - dodatkowo zapisuje model na dysk i loguje go jako artefakt W&B.
    """
    global wandb_run

    log.info("Zapisywanie obiektu TabularPredictor jako 'ag_model'.")

    # --- Ręczny zapis modelu do pliku, który potem stanie się artefaktem W&B ---
    model_path = Path("data/06_models/ag_production.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # ruff: noqa: PLC0415
    try:
        import joblib

        joblib.dump(ag_predictor, model_path)
        log.info(f"Zapisano model AutoGluon do pliku: {model_path}")
    except Exception as e:
        log.warning(f"Nie udało się zapisać modelu lokalnie: {e}")

    # --- Logowanie modelu jako artefakt W&B z aliasem 'candidate' ---
    if wandb_run is not None and model_path.exists():
        try:
            art = wandb.Artifact("ag_model", type="model")
            art.add_file(str(model_path))
            wandb_run.log_artifact(art, aliases=["candidate"])
            log.info("Zalogowano artefakt modelu do W&B (ag_model:candidate).")
        except Exception as e:
            log.warning(f"Nie udało się zalogować artefaktu modelu do W&B: {e}")
        finally:
            try:
                wandb_run.finish()
            except Exception:
                pass

    # Kedro dalej zapisze ten sam obiekt do ag_model (PickleDataSet)
    return ag_predictor


def save_prediction_to_db(
    ag_metrics: dict,
    X_test: pd.DataFrame,
    ag_predictor: TabularPredictor,
    autogluon_params: dict,
) -> dict:

    if ag_metrics is None:
        ag_metrics = {}

    payload = {
        "n_test_samples": len(X_test),
        "n_features": len(X_test.columns),
        "model_type": "autogluon",
        "problem_type": autogluon_params.get("problem_type", "unknown"),
    }

    main_metric_key = f"ag_test_score_{ag_predictor.eval_metric}"
    prediction_value = ag_metrics.get(main_metric_key, 0.0)

    model_version = autogluon_params.get("label", "ag_model:v1")

    try:
        record_id = save_prediction(
            payload=payload,
            prediction=prediction_value,
            model_version=model_version,
        )
        log.info(f"Metryki zapisane do bazy z ID: {record_id}")
        return {
            "status": "success",
            "record_id": record_id,
            "model_version": model_version,
            "main_metric": prediction_value,
        }
    except Exception as e:
        log.error(f"Nie udało się zapisać do bazy: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
