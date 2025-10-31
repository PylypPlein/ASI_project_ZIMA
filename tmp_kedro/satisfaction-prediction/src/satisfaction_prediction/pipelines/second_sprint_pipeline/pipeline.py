from kedro.pipeline import Pipeline, node

from satisfaction_prediction.pipelines.second_sprint_pipeline.nodes import (
    basic_clean,
    evaluate,
    load_raw,
    split_data,
    train_baseline,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
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
                inputs=["clean_data", "params:target_col"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_baseline,
                inputs=["X_train", "y_train"],
                outputs="baseline_model",
                name="train_baseline_node",
            ),
            node(
                func=evaluate,
                inputs=["baseline_model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_node",
            ),
        ]
    )
