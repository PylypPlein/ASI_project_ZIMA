from kedro.pipeline import Pipeline, node
from .nodes import (
    # Nody do przetwarzania danych
    load_raw,
    basic_clean,
    split_data,
    # Nody AutoGluon (Zadanie 2)
    train_autogluon,
    evaluate_autogluon,
    save_best_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Tworzy pipeline dla przetwarzania danych i trenowania modelu AutoGluon.
    """
    return Pipeline(
        [
            # 1. PRZETWARZANIE DANYCH
            node(
                func=load_raw,
                inputs="params:data_path",
                outputs="raw_data",
                name="load_raw_node",
            ),
            node(
                func=basic_clean,
                inputs="raw_data",
                outputs="clean_data",
                name="basic_clean_node",
            ),
            node(
                func=split_data,
                # params:run_params jest dodane, aby przekazać random_state do split_data
                inputs=["clean_data", "params:target_col", "params:run_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),

            # 2. TRENING I EWALUACJA AUTOGLUON (po split)
            node(
                func=train_autogluon,
                inputs=["X_train", "y_train", "params"],  # 'params' zawiera parametry dla AutoGluon
                outputs="ag_predictor",
                name="train_autogluon_node",
            ),
            node(
                func=evaluate_autogluon,
                inputs=["ag_predictor", "X_test", "y_test"],
                outputs="ag_metrics",  # Zapis do catalog.yml
                name="evaluate_autogluon_node",
            ),
            node(
                func=save_best_model,
                inputs=["ag_predictor"],
                outputs="ag_model",  # Zapis do catalog.yml
                name="save_best_model_node",
            ),
        ]
    )