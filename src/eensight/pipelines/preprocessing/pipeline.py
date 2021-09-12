# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
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
            inputs=["root_input", "rebind_names", "location"],
            outputs="validated_data",
            name="validate_input_data",
        ),
        node(
            func=find_outliers,
            inputs=["validated_data", "parameters"],
            outputs="data_with_outliers",
            name="find_outliers",
        ),
        node(
            func=outlier_to_nan,
            inputs="data_with_outliers",
            outputs="data_without_outliers",
            name="outlier_to_nan",
        ),
        node(
            func=linear_inpute_missing,
            inputs=["data_without_outliers", "parameters"],
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
            inputs="preprocessed_data",
            outputs="data_adequacy_summary",
            name="check_data_adequacy",
        ),
    ]
)

preprocess_train = pipeline(
    preprocess,
    inputs=["rebind_names", "location"],  # don't namespace
    outputs="data_adequacy_summary",  # don't namespace
    namespace="train",
)

preprocess_test = pipeline(
    pipeline(
        preprocess.only_nodes(
            "validate_input_data", "find_outliers", "drop_missing_data"
        ),
        outputs={"data_with_outliers": "preprocessed_data"},
    ),
    inputs=["rebind_names", "location"],  # don't namespace
    namespace="test",
)


def create_pipeline(**kwargs):
    return Pipeline([preprocess_train], tags="train") + Pipeline(
        [preprocess_test], tags="test"
    )
