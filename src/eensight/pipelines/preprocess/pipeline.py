# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    apply_drift_detection,
    check_data_adequacy,
    drop_missing_data,
    find_outliers,
    linear_inpute_missing,
    outlier_to_nan,
    validate_input_data,
)

preprocess = Pipeline(
    [
        node(
            func=validate_input_data,
            inputs=["root_input", "rebind_names"],
            outputs="validated_data",
            name="validate_input_data",
        ),
        node(
            func=apply_drift_detection,
            inputs=["validated_data", "params:of_drift_detection"],
            outputs=["data_with_drift", "data_no_drift"],
            name="apply_drift_detection",
        ),
        node(
            func=find_outliers,
            inputs=[
                "data_no_drift",
                "params:find_outliers_for",
                "params:of_global_filter",
                "params:of_seasonal_decompose",
                "params:of_global_outlier",
                "params:of_local_outlier",
                "params:max_outlier_pct",
            ],
            outputs="data_with_outliers",
            name="find_outliers",
        ),
        node(
            func=outlier_to_nan,
            inputs="data_with_outliers",
            outputs="data_cleaned",
            name="outlier_to_nan",
        ),
        node(
            func=linear_inpute_missing,
            inputs=["data_cleaned", "params:of_linear_impute"],
            outputs="preprocessed_data",
            name="linear_inpute_missing",
        ),
        node(
            func=drop_missing_data,
            inputs="preprocessed_data",
            outputs="model_input_data",
            name="drop_missing_data",
        ),
        node(
            func=check_data_adequacy,
            inputs=["preprocessed_data", "params:max_missing_pct"],
            outputs="data_adequacy",
            name="check_data_adequacy",
        ),
    ]
)

preprocess_train = pipeline(
    preprocess,
    inputs="rebind_names",  # don't namespace
    namespace="train",
)

preprocess_apply = pipeline(
    pipeline(
        preprocess.only_nodes(
            "validate_input_data",
            "linear_inpute_missing",
            "drop_missing_data",
        ),
        outputs={"validated_data": "data_cleaned"},
    ),
    inputs="rebind_names",  # don't namespace
    namespace="apply",
)


def create_pipeline(**kwargs):
    return Pipeline([preprocess_train], tags="train") + Pipeline(
        [preprocess_apply], tags="apply"
    )
