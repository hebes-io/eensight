# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from eensight.models import CalendarEnsemble, ParetoEnsemble

from .cross_validation import CrossValidator
from .metrics import cvrmse, nmbe
from .optimization import optimize

logger = logging.getLogger("model-selection-stage")


def split_training_data(data, drop_outliers=False):
    data = data.dropna()
    outliers = data.pop("consumption_outlier")
    if drop_outliers:
        X_train = data.drop("consumption", axis=1).loc[~outliers]
        y_train = data.loc[~outliers, ["consumption"]]
    else:
        X_train = data.drop("consumption", axis=1)
        y_train = data[["consumption"]]
    return X_train, y_train


def optimize_model(data, model_structure, parameters):
    params = parameters["optimize_model"]
    model = CalendarEnsemble(
        model_structure=model_structure,
        weight_method=params.get("weight_method", "softmin"),
        cache_location=params.get("cache_location"),
        alpha=params.get("alpha", 0.1),
        fit_intercept=params.get("fit_intercept", False),
    )

    X_train, y_train = split_training_data(data, drop_outliers=True)
    res = optimize(
        model,
        X_train,
        y_train,
        budget=params.get("budget", 20),
        test_size=params.get("test_size", 0.2),
        n_repeats=params.get("n_repeats", 2),
        out_of_sample=params.get("out_of_sample", True),
        multivariate=True,
    )
    return model, res.scores, res.params


def create_pareto_ensemble(data, model, opt_scores, opt_params, parameters):
    to_include = opt_scores.sort_values(by="CVRMSE").iloc[:5]
    model_ens = ParetoEnsemble(
        base_estimator=model,
        ensemble_parameters=opt_params.loc[to_include.index].to_dict("records"),
    )

    X_train, y_train = split_training_data(data, drop_outliers=True)
    model_ens = model_ens.fit(X_train, y_train)

    params = parameters["create_pareto_ensemble"]
    if params["evaluate_with_outliers"]:
        X_eval, y_eval = split_training_data(data, drop_outliers=False)
        pred = model_ens.predict(
            X_eval, include_components=params.get("include_components", False)
        )
        logger.info(
            "In-sample CV(RMSE) with outliers (%): "
            f"{cvrmse(y_eval['consumption'], pred['consumption'])*100}"
        )
        logger.info(
            "In-sample NMBE with outliers (%): "
            f"{nmbe(y_eval['consumption'], pred['consumption'])*100}"
        )
    else:
        pred = model_ens.predict(
            X_train, include_components=params.get("include_components", False)
        )
        logger.info(
            "In-sample CV(RMSE) without outliers (%): "
            f"{cvrmse(y_train['consumption'], pred['consumption'])*100}"
        )
        logger.info(
            "In-sample NMBE without outliers (%): "
            f"{nmbe(y_train['consumption'], pred['consumption'])*100}"
        )
    logger.info(f"Number of parameters: {model_ens.n_parameters}")
    return model_ens, pred


def cross_validate(data, model, parameters):
    params = parameters["cross_validate_model"]
    X_train, y_train = split_training_data(
        data, drop_outliers=not params.get("evaluate_with_outliers", True)
    )
    cv_model = CrossValidator(
        model,
        group_by=params.get("group_by", "week"),
        stratify_by=params.get("stratify_by", "month"),
        n_splits=params.get("n_splits", 3),
        n_repeats=params.get("n_repeats", 5),
        keep_estimators=params.get("keep_estimators", True),
        n_jobs=params.get("n_jobs", -1),
    )
    cv_model = cv_model.fit(X_train, y_train)
    logger.info(
        f'Average out-of-sample CVRMSE (%): {cv_model.scores_["CVRMSE"].mean()*100}'
    )
    logger.info(
        f'Average out-of-sample NMBE: (%) {cv_model.scores_["NMBE"].mean()*100}'
    )
    return cv_model
