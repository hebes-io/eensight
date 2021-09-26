# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
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

from .cross_validation import CrossValidator
from .metrics import cvrmse, nmbe
from .optimization import optimize

logger = logging.getLogger("model-selection")


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
    timeout,
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
        timeout=timeout,
        test_size=test_size,
        n_repeats=n_repeats,
        out_of_sample=out_of_sample,
        multivariate=multivariate,
        tags=tag,
    )


def optimize_model(data, model_structure, distance_metrics, for_optimize_model):
    mcc_features = Pipeline(
        [
            ("dates", DatetimeFeatures(subset=["month", "dayofweek"])),
            ("features", MMCFeatures()),
        ]
    )
    clusterer = ClusterFeatures(
        min_samples=for_optimize_model["clusterer"].get("min_samples", 5),
        transformer=mcc_features,
        n_neighbors=for_optimize_model["clusterer"].get("n_neighbors", 1),
        weights=for_optimize_model["clusterer"].get("weights", "uniform"),
        output_name=for_optimize_model.get("group_feature", "cluster"),
    )
    reg_grouped = GroupedPredictor(
        model_structure=model_structure,
        group_feature=for_optimize_model.get("group_feature", "cluster"),
        estimator_params=(
            ("alpha", for_optimize_model["grouped"].get("alpha", 0.01)),
            ("fit_intercept", False),
        ),
    )
    model = CompositePredictor(
        distance_metrics=distance_metrics,
        base_clusterer=clusterer,
        base_regressor=reg_grouped,
        group_feature=for_optimize_model.get("group_feature", "cluster"),
    )

    X_train, y_train, _ = split_training_data(data)
    parallel = Parallel(n_jobs=2)
    results = parallel(
        delayed(optimize_for_tag)(
            clone(model),
            X_train,
            y_train,
            budget=for_optimize_model.get("budget", 15),
            timeout=for_optimize_model.get("timeout"),
            test_size=for_optimize_model.get("test_size", 0.25),
            n_repeats=for_optimize_model.get("n_repeats", 1),
            out_of_sample=for_optimize_model.get("out_of_sample", True),
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


def create_ensemble(data, model, opt_scores, opt_params):
    to_include = opt_scores.sort_values(by="CVRMSE").iloc[:5]

    model_ens = EnsemblePredictor(
        base_estimator=model,
        ensemble_parameters=opt_params.loc[to_include.index].to_dict("records"),
    )
    X_train, y_train, _ = split_training_data(data)
    model_ens = model_ens.fit(X_train, y_train)
    return model_ens


def evaluate_ensemble(data, model, include_components, for_cross_validate):
    X_train, y_train, _ = split_training_data(data)
    pred = model.predict(X_train, include_components=include_components)
    logger.info(
        "In-sample CV(RMSE) (%): "
        f"{cvrmse(y_train['consumption'], pred['consumption'])*100}"
    )
    logger.info(
        "In-sample NMBE (%): "
        f"{nmbe(y_train['consumption'], pred['consumption'])*100}"
    )
    logger.info(f"Number of parameters: {model.n_parameters}")
    logger.info(f"Degrees of freedom: {model.dof}")

    cv = CrossValidator(
        model,
        group_by=for_cross_validate.get("group_by", "week"),
        stratify_by=for_cross_validate.get("stratify_by", "month"),
        n_splits=for_cross_validate.get("n_splits", 3),
        n_repeats=for_cross_validate.get("n_repeats", 5),
        n_jobs=for_cross_validate.get("n_jobs", -1),
        keep_estimators=True,
        verbose=True,
    )

    if for_cross_validate.get("do_cross_validate", False):
        cv = cv.fit(X_train, y_train)
        logger.info(
            f'Mean cross-validation CVRMSE (%): {np.mean(cv.scores_["CVRMSE"])*100}'
        )
        logger.info(
            f'Mean cross-validation NMBE: (%) {np.mean(cv.scores_["NMBE"])*100}'
        )

    return pred, cv


def apply_ensemble(data, model, include_components):
    X_test, y_test, outliers = split_training_data(data, drop_outliers=False)

    other_outliers = X_test.filter(like="outlier", axis=1)
    if not other_outliers.empty:
        X_test = X_test.drop(other_outliers.columns, axis=1)
        outliers = np.logical_or(outliers, other_outliers.apply(any, axis=1))

    pred = model.predict(X_test, include_components=include_components)

    y_true = y_test["consumption"]
    y_pred = pred["consumption"]
    logger.info(
        "Out-of-sample CV(RMSE) with outliers (%): " f"{cvrmse(y_true, y_pred)*100}"
    )
    logger.info("Out-of-sample NMBE with outliers (%): " f"{nmbe(y_true, y_pred)*100}")

    y_true = y_test.loc[~outliers, ["consumption"]]
    y_pred = pred.loc[~outliers, ["consumption"]]
    logger.info(
        "Out-of-sample CV(RMSE) without outliers (%): " f"{cvrmse(y_true, y_pred)*100}"
    )
    logger.info(
        "Out-of-sample NMBE without outliers (%): " f"{nmbe(y_true, y_pred)*100}"
    )
    return pred
