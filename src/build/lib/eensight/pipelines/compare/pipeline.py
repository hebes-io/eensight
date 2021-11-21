# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import apply_compare

compare = Pipeline(
    [
        node(
            func=apply_compare,
            inputs=[
                "model_input_data",
                "mean_model",
                "conformal_model",
                "params:of_apply_compare",
            ],
            outputs="savings",
            name="apply_compare",
        ),
    ]
)


compare_apply = pipeline(
    compare,
    inputs=["mean_model", "conformal_model"],  # don't namespace
    namespace="apply",
)


def create_pipeline(**kwargs):
    return Pipeline([compare_apply], tags="apply")
