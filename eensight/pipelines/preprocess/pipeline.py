# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_inputs, validate_inputs

preprocess_ppl = Pipeline(
    [
        node(
            func=validate_inputs,
            inputs=[
                "input-features",
                "input-labels",
                "params:rebind_names",
                "params:date_format",
                "params:validation.threshold",
                "params:alignment.mode",
                "params:alignment.tolerance",
                "params:alignment.cumulative",
            ],
            outputs=["validated-features", "validated-labels"],
            name="validate_inputs",
        ),
        node(
            func=evaluate_inputs,
            inputs=[
                "validated-features",
                "validated-labels",
                "params:filter.min_value",
                "params:filter.max_value",
                "params:filter.allow_zero",
                "params:filter.allow_negative",
                "params:adequacy.max_missing_pct",
            ],
            outputs=[
                "preprocessed-features",
                "preprocessed-labels",
                "adequacy-summary",
            ],
            name="evaluate_inputs",
        ),
    ]
)

preprocess_train = pipeline(
    preprocess_ppl,
    parameters=[
        "rebind_names",
        "date_format",
        "validation.threshold",
        "alignment.mode",
        "alignment.tolerance",
        "alignment.cumulative",
        "filter.min_value",
        "filter.max_value",
        "filter.allow_zero",
        "filter.allow_negative",
        "adequacy.max_missing_pct",
    ],
    namespace="train",
)

preprocess_test = pipeline(
    pipeline(
        preprocess_ppl.only_nodes("validate_inputs"),
        outputs={
            "validated-features": "preprocessed-features",
            "validated-labels": "preprocessed-labels",
        },
    ),
    parameters=[
        "rebind_names",
        "date_format",
        "validation.threshold",
        "alignment.mode",
        "alignment.tolerance",
        "alignment.cumulative",
    ],
    namespace="test",
)


preprocess_apply = pipeline(
    pipeline(
        preprocess_ppl.only_nodes("validate_inputs"),
        outputs={
            "validated-features": "preprocessed-features",
            "validated-labels": "preprocessed-labels",
        },
    ),
    parameters=[
        "rebind_names",
        "date_format",
        "validation.threshold",
        "alignment.mode",
        "alignment.tolerance",
        "alignment.cumulative",
    ],
    namespace="apply",
)


def create_pipeline(**kwargs):
    return (
        Pipeline([preprocess_train], tags=["train"])
        + Pipeline([preprocess_test], tags=["test"])
        + Pipeline([preprocess_apply], tags=["apply"])
    )
