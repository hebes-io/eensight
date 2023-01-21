# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import adjust_activity_levels

ppl_nodes = {
    "adjust_activity_levels": node(
        func=adjust_activity_levels,
        inputs=[
            "preprocessed-features",
            "preprocessed-labels",
            "model-autoenc",
            "params:activity.non_occ_features",
            "params:activity.cat_features",
            "params:activity.assume_hurdle",
            "params:activity.n_trials_adjust",
            "params:activity.upper_bound",
            "params:activity.verbose",
        ],
        outputs="activity-adjusted",
        name="adjust_activity_levels",
    ),
}

adjust_apply = pipeline(
    pipeline([ppl_nodes["adjust_activity_levels"]]),
    inputs=["model-autoenc"],
    parameters=[
        "activity.non_occ_features",
        "activity.cat_features",
        "activity.assume_hurdle",
        "activity.n_trials_adjust",
        "activity.upper_bound",
        "activity.verbose",
    ],
    namespace="apply",
)


def create_pipeline(**kwargs):
    return Pipeline([adjust_apply], tags=["apply_autoenc"])
