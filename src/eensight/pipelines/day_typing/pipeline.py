# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import apply_day_typing, get_daily_matrix_profile, prepare_data

daytype = Pipeline(
    [
        node(
            func=prepare_data,
            inputs=["preprocessed_data", "parameters"],
            outputs="data_for_daytyping",
            name="prepare_data",
        ),
        node(
            func=get_daily_matrix_profile,
            inputs="data_for_daytyping",
            outputs="matrix_profile_scores",
            name="get_daily_matrix_profile",
        ),
        node(
            func=apply_day_typing,
            inputs=["data_for_daytyping", "parameters"],
            outputs="distance_metrics",
            name="apply_day_typing",
        ),
    ]
)


daytype_train = pipeline(
    daytype,
    outputs=["matrix_profile_scores", "distance_metrics"],  # don't namespace
    namespace="train",
)


def create_pipeline(**kwargs):
    return Pipeline([daytype_train], tags="train")
