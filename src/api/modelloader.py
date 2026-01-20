# src/api/modelloader.py

import os
from functools import lru_cache
from typing import Tuple

from autogluon.tabular import TabularPredictor

# Ścieżka do katalogu z modelem AutoGluon
DEFAULT_MODEL_DIR = (
    "tmp_kedro/data/06_models/autogluon_temp_output"
)

# Wersja modelu
DEFAULT_MODEL_VERSION = "ag_model:v2"


@lru_cache(maxsize=1)
def get_model() -> Tuple[TabularPredictor, str]:
    """
    Ładuje model 'Production' z lokalnego katalogu AutoGluon jako singleton.

    Ścieżka może być nadpisana zmienną środowiskową MODEL_DIR.
    Zwraca:
        (predictor, model_version)
    """
    model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
    model_version = os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"MODEL_DIR does not exist: {model_dir}. "
            "Expected AutoGluon model directory "
            "tmp_kedro/satisfaction-prediction/data/06_models/autogluon_temp_output "
            "or override MODEL_DIR env variable."
        )

    predictor: TabularPredictor = TabularPredictor.load(model_dir)
    return predictor, model_version
