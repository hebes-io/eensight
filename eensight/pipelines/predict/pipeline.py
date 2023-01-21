# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_activity_feature,
    fit_predict,
    fit_predict_autoenc,
    predict,
    predict_autoenc,
)

ppl_nodes = {
    "create_activity_feature": node(
        func=create_activity_feature,
        inputs=[
            "preprocessed-features",
            "preprocessed-labels",
            "params:activity.non_occ_features",
            "params:activity.cat_features",
            "params:activity.exog",
            "params:activity.assume_hurdle",
            "params:activity.n_trials",
            "params:activity.verbose",
            "params:activity.adjusted_activity",
        ],
        outputs="activity",
        name="create_activity_feature",
    ),
    "fit_predict": node(
        func=fit_predict,
        inputs=[
            "preprocessed-features",
            "preprocessed-labels",
            "params:fit.lags",
            "params:fit.cat_features",
            "params:fit.validation_size",
        ],
        outputs=["model", "prediction", "performance"],
        name="fit_predict",
    ),
    "fit_predict_autoenc": node(
        func=fit_predict_autoenc,
        inputs=[
            "preprocessed-features",
            "preprocessed-labels",
            "activity",
            "params:fit.lags",
            "params:fit.cat_features",
            "params:fit.validation_size",
            "params:activity.assume_hurdle",
        ],
        outputs=["model-autoenc", "prediction-autoenc", "performance-autoenc"],
        name="fit_predict_autoenc",
    ),
    "predict": node(
        func=predict,
        inputs=["preprocessed-features", "model"],
        outputs="prediction",
        name="predict",
    ),
    "predict_autoenc": node(
        func=predict_autoenc,
        inputs=[
            "preprocessed-features",
            "activity",
            "model-autoenc",
            "params:activity.assume_hurdle",
        ],
        outputs="prediction-autoenc",
        name="predict_autoenc",
    ),
}


predict_train = pipeline(
    pipeline([ppl_nodes["fit_predict"]]),
    outputs=["model"],
    parameters=[
        "fit.lags",
        "fit.cat_features",
        "fit.validation_size",
    ],
    namespace="train",
)


predict_train_autoenc = pipeline(
    pipeline(
        [
            ppl_nodes["create_activity_feature"],
            ppl_nodes["fit_predict_autoenc"],
        ],
    ),
    outputs=["model-autoenc"],
    parameters=[
        "fit.lags",
        "fit.cat_features",
        "fit.validation_size",
        "activity.non_occ_features",
        "activity.cat_features",
        "activity.exog",
        "activity.assume_hurdle",
        "activity.n_trials",
        "activity.verbose",
        "activity.adjusted_activity",
    ],
    namespace="train",
)


predict_test = pipeline(
    pipeline([ppl_nodes["predict"]]),
    inputs=["model"],
    namespace="test",
)


predict_test_autoenc = pipeline(
    pipeline(
        [
            ppl_nodes["create_activity_feature"],
            ppl_nodes["predict_autoenc"],
        ],
    ),
    inputs=["model-autoenc"],
    parameters=[
        "activity.non_occ_features",
        "activity.cat_features",
        "activity.exog",
        "activity.assume_hurdle",
        "activity.n_trials",
        "activity.verbose",
        "activity.adjusted_activity",
    ],
    namespace="test",
)


predict_apply = pipeline(
    pipeline([ppl_nodes["predict"]]),
    inputs=["model"],
    namespace="apply",
)


predict_apply_autoenc = pipeline(
    pipeline(
        [
            ppl_nodes["create_activity_feature"],
            ppl_nodes["predict_autoenc"],
        ],
    ),
    inputs=["model-autoenc"],
    parameters=[
        "activity.non_occ_features",
        "activity.cat_features",
        "activity.exog",
        "activity.assume_hurdle",
        "activity.n_trials",
        "activity.verbose",
        "activity.adjusted_activity",
    ],
    namespace="apply",
)


def create_pipeline(**kwargs):
    return (
        Pipeline([predict_train], tags=["train_default"])
        + Pipeline([predict_test], tags=["test_default"])
        + Pipeline([predict_apply], tags=["apply_default"])
        + Pipeline([predict_train_autoenc], tags=["train_autoenc"])
        + Pipeline([predict_test_autoenc], tags=["test_autoenc"])
        + Pipeline([predict_apply_autoenc], tags=["apply_autoenc"])
    )
