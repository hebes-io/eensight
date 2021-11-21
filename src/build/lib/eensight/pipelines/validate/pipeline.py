# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_conformal, cross_validate

validate = Pipeline(
    [
        node(
            func=cross_validate,
            inputs=["model_input_data", "mean_model", "params:of_cross_validate"],
            outputs="cv_model",
            name="cross_validate",
        ),
        node(
            func=create_conformal,
            inputs=["model_input_data", "cv_model", "params:of_conformal_predictor"],
            outputs="conformal_model",
            name="create_conformal_predictor",
        ),
    ]
)

validate_train = pipeline(
    validate,
    inputs="mean_model",  # don't namespace
    outputs=["cv_model", "conformal_model"],
    namespace="train",
)


def create_pipeline(**kwargs):
    return Pipeline([validate_train], tags="train")
