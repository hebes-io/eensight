# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import pandas as pd
from joblib import Parallel
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import delayed

from eensight.features.cluster import ClusterFeatures
from eensight.features.generate import DatetimeFeatures, MMCFeatures
from eensight.models import CompositePredictor, EnsemblePredictor, GroupedPredictor

from .metrics import cvrmse, nmbe
from .optimization import optimize

logger = logging.getLogger("model-selection-stage")


def split_training_data(data, drop_outliers=True):
    outliers = data.pop("consumption_outlier")
    if drop_outliers:
        X_train = data.drop("consumption", axis=1).loc[~outliers]
    else:
        X_train = data.drop("consumption", axis=1)

    y_train = data.loc[X_train.index, ["consumption"]]
    return X_train, y_train, outliers


def optimize_for_tag(
    model,
    X_train,
    y_train,
    *,
    budget,
    test_size,
    n_repeats,
    out_of_sample,
    multivariate,
    tag,
):
    model.set_params(
        **{"base_clusterer__assign_clusters__cluster_selection_method": tag}
    )
    return optimize(
        model,
        X_train,
        y_train,
        budget=budget,
        test_size=test_size,
        n_repeats=n_repeats,
        out_of_sample=out_of_sample,
        multivariate=multivariate,
        tags=tag,
    )


def optimize_model(data, model_structure, distance_metrics, parameters):
    params = parameters["optimize_model"]

    mcc_features = Pipeline(
        [
            ("dates", DatetimeFeatures(subset=["month", "dayofweek"])),
            ("features", MMCFeatures()),
        ]
    )
    clusterer = ClusterFeatures(
        min_samples=params["clusterer"].get("min_samples", 5),
        transformer=mcc_features,
        n_neighbors=params["clusterer"].get("n_neighbors", 1),
        weights=params["clusterer"].get("weights", "uniform"),
        output_name=params.get("group_feature", "cluster"),
    )
    reg_grouped = GroupedPredictor(
        model_structure=model_structure,
        group_feature=params.get("group_feature", "cluster"),
        estimator_params=(
            ("alpha", params["grouped"].get("alpha", 0.01)),
            ("fit_intercept", False),
        ),
    )
    model = CompositePredictor(
        distance_metrics=distance_metrics,
        base_clusterer=clusterer,
        base_regressor=reg_grouped,
        group_feature=params.get("group_feature", "cluster"),
    )

    X_train, y_train, _ = split_training_data(data)
    parallel = Parallel(n_jobs=2)
    results = parallel(
        delayed(optimize_for_tag)(
            clone(model),
            X_train,
            y_train,
            budget=params.get("budget", 15),
            test_size=params.get("test_size", 0.25),
            n_repeats=params.get("n_repeats", 1),
            out_of_sample=params.get("out_of_sample", True),
            multivariate=False,
            tag=tag,
        )
        for tag in ["eom", "leaf"]
    )

    scores = None
    params = None
    for res in results:
        scores = pd.concat((scores, res["scores"]), ignore_index=True)
        params = pd.concat(
            (
                params,
                res["params"].assign(
                    **{"assign_clusters__cluster_selection_method": res["tags"]}
                ),
            ),
            ignore_index=True,
        )
    logger.info(f"Optimization scores: {scores.to_dict()}")
    return model, scores, params


def create_ensemble(data, model, opt_scores, opt_params, parameters):
    to_include = opt_scores.sort_values(by="CVRMSE").iloc[:5]

    model_ens = EnsemblePredictor(
        base_estimator=model,
        ensemble_parameters=opt_params.loc[to_include.index].to_dict("records"),
    )

    X_train, y_train, _ = split_training_data(data)
    model_ens = model_ens.fit(X_train, y_train)

    params = parameters["create_ensemble"]
    pred = model_ens.predict(
        X_train, include_components=params.get("include_components", True)
    )
    logger.info(
        "In-sample CV(RMSE) (%): "
        f"{cvrmse(y_train['consumption'], pred['consumption'])*100}"
    )
    logger.info(
        "In-sample NMBE (%): "
        f"{nmbe(y_train['consumption'], pred['consumption'])*100}"
    )
    logger.info(f"Number of parameters: {model_ens.n_parameters}")
    return model_ens, pred


def apply_ensemble(data, model_ens, parameters):
    params = parameters["apply_ensemble"]
    X_test, y_test, outliers = split_training_data(data, drop_outliers=False)

    other_outliers = X_test.filter(like="outlier", axis=1)
    if not other_outliers.empty:
        X_test = X_test.drop(other_outliers.columns, axis=1)
        outliers = np.logical_or(outliers, other_outliers.apply(any, axis=1))

    pred = model_ens.predict(
        X_test, include_components=params.get("include_components", True)
    )

    y_true = y_test["consumption"]
    y_pred = pred["consumption"]
    logger.info(
        "Out-of-sample CV(RMSE) with outliers (%): " f"{cvrmse(y_true, y_pred)*100}"
    )

    y_true = y_test.loc[~outliers, ["consumption"]]
    y_pred = pred.loc[~outliers, ["consumption"]]
    logger.info(
        "Out-of-sample CV(RMSE) without outliers (%): " f"{cvrmse(y_true, y_pred)*100}"
    )
    return pred
