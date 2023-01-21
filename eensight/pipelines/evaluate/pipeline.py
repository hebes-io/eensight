# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_on_apply,
    evaluate_on_apply_autoenc,
    evaluate_on_test,
    evaluate_on_test_autoenc,
)

ppl_nodes = {
    "evaluate_on_test": node(
        func=evaluate_on_test,
        inputs=["preprocessed-labels", "prediction"],
        outputs="performance",
        name="evaluate_on_test",
    ),
    "evaluate_on_test_autoenc": node(
        func=evaluate_on_test_autoenc,
        inputs=[
            "preprocessed-labels",
            "prediction-autoenc",
            "activity",
            "params:activity.assume_hurdle",
        ],
        outputs="performance-autoenc",
        name="evaluate_on_test_autoenc",
    ),
    "evaluate_on_apply": node(
        func=evaluate_on_apply,
        inputs=["preprocessed-labels", "prediction"],
        outputs="impact",
        name="evaluate_on_apply",
    ),
    "evaluate_on_apply_autoenc": node(
        func=evaluate_on_apply_autoenc,
        inputs=[
            "preprocessed-labels",
            "prediction-autoenc",
            "activity",
            "params:activity.assume_hurdle",
        ],
        outputs="impact-autoenc",
        name="evaluate_on_apply_autoenc",
    ),
}


evaluate_test = pipeline(
    pipeline([ppl_nodes["evaluate_on_test"]]),
    namespace="test",
)


evaluate_test_autoenc = pipeline(
    pipeline([ppl_nodes["evaluate_on_test_autoenc"]]),
    parameters=["activity.assume_hurdle"],
    namespace="test",
)

evaluate_apply = pipeline(
    pipeline([ppl_nodes["evaluate_on_apply"]]),
    namespace="apply",
)

evaluate_apply_autoenc = pipeline(
    pipeline([ppl_nodes["evaluate_on_apply_autoenc"]]),
    parameters=["activity.assume_hurdle"],
    namespace="apply",
)


def create_pipeline(**kwargs):
    return (
        Pipeline([evaluate_test], tags=["test_default"])
        + Pipeline([evaluate_test_autoenc], tags=["test_autoenc"])
        + Pipeline([evaluate_apply], tags=["apply_default"])
        + Pipeline([evaluate_apply_autoenc], tags=["apply_autoenc"])
    )
