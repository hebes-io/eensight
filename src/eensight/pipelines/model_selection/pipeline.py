# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_pareto_ensemble, cross_validate, optimize_model

baseline = Pipeline(
    [
        node(
            func=optimize_model,
            inputs=["preprocessed_data", "model_structure", "parameters"],
            outputs=["base_model", "opt_scores", "opt_params"],
            name="optimize_model",
        ),
        node(
            func=create_pareto_ensemble,
            inputs=[
                "preprocessed_data",
                "base_model",
                "opt_scores",
                "opt_params",
                "parameters",
            ],
            outputs=["pareto_ensemble_model", "pareto_ensemble_prediction"],
            name="create_pareto_ensemble",
        ),
        node(
            func=cross_validate,
            inputs=["preprocessed_data", "pareto_ensemble_model", "parameters"],
            outputs="cross_validator_model",
            name="cross_validate",
        ),
    ]
)

baseline_train = pipeline(
    baseline,
    inputs="model_structure",  # don't namespace
    namespace="train",
)


def create_pipeline(**kwargs):
    return Pipeline([baseline_train], tags="train")
