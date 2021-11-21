# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fit_model, optimize_model

baseline = Pipeline(
    [
        node(
            func=optimize_model,
            inputs=[
                "model_input_data",
                "location",
                "model_config",
                "feature_map",
                "distance_metrics",
                "params:of_optimize_model",
            ],
            outputs=["opt_params", "optimized_model"],
            name="optimize_model",
        ),
        node(
            func=fit_model,
            inputs=[
                "model_input_data",
                "optimized_model",
            ],
            outputs="mean_model",
            name="fit_model",
        ),
    ]
)


baseline_train = pipeline(
    baseline,
    inputs=[
        "location",
        "model_config",
        "feature_map",
    ],
    outputs=["opt_params", "mean_model"],
    namespace="train",
)


def create_pipeline(**kwargs):
    return Pipeline([baseline_train], tags="train")
