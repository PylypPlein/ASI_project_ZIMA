from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from satisfaction_prediction.pipelines.second_sprint_pipeline.nodes import (
    evaluate_autogluon,
    train_autogluon,
)


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(123)

    X_train = pd.DataFrame(
        {
            "feat1": rng.normal(size=80),
            "feat2": rng.integers(0, 10, size=80),
        }
    )
    y_train = pd.Series(rng.integers(0, 2, size=80), name="satisfaction")

    X_test = pd.DataFrame(
        {
            "feat1": rng.normal(size=20),
            "feat2": rng.integers(0, 10, size=20),
        }
    )
    y_test = pd.Series(rng.integers(0, 2, size=20), name="satisfaction")

    return X_train, y_train, X_test, y_test


@pytest.fixture
def autogluon_params():
    return {
        "label": "satisfaction",
        "problem_type": "binary",
        "eval_metric": "roc_auc",
        "time_limit": 5,
        "presets": "medium_quality_faster_train",
        "random_state": 42,
    }


def test_evaluate_autogluon_returns_expected_metrics(
    mocker, small_dataset, autogluon_params
):
    mocker.patch(
        "satisfaction_prediction.pipelines.second_sprint_pipeline.nodes.wandb.init",
        return_value=None,
    )
    mocker.patch(
        "satisfaction_prediction.pipelines.second_sprint_pipeline.nodes.wandb.init",
        return_value=None,
    )
    mocker.patch(
        "satisfaction_prediction.pipelines.second_sprint_pipeline.nodes.wandb.Artifact"
    )

    X_train, y_train, X_test, y_test = small_dataset

    predictor = train_autogluon(X_train, y_train, autogluon_params)
    metrics = evaluate_autogluon(predictor, X_test, y_test)

    assert isinstance(metrics, dict), "evaluate_autogluon powinien zwracać dict"

    expected_keys = {
        f"ag_test_score_{predictor.eval_metric}",
        "ag_test_error_rate",
        "ag_roc_auc",
    }
    assert expected_keys.issubset(
        metrics.keys()
    ), f"Brakuje oczekiwanych kluczy metryk: {expected_keys - set(metrics.keys())}"

    main_score_key = f"ag_test_score_{predictor.eval_metric}"

    if metrics[main_score_key] is not None:
        assert 0.0 <= metrics[main_score_key] <= 1.0

    if metrics["ag_roc_auc"] is not None:
        assert 0.0 <= metrics["ag_roc_auc"] <= 1.0

    assert 0.0 <= metrics["ag_test_error_rate"] <= 1.0


def test_models_directory_exists():
    models_dir = Path("data/06_models")
    assert models_dir.exists() and models_dir.is_dir(), "Brak katalogu data/06_models/"


def test_production_model_file_exists():
    model_path = Path("data/06_models/ag_production.pkl")
    assert model_path.exists(), "Brak pliku modelu: data/06_models/ag_production.pkl"


def test_metrics_file_exists():
    metrics_path = Path("data/09_tracking/ag_metrics.json")
    assert metrics_path.exists(), "Brak pliku metryk: data/09_tracking/ag_metrics.json"
