# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import pandas as pd

from eensight.models import AggregatePredictor
from eensight.utils import split_training_data

from .conformal import AggregatedCp
from .cross_validation import CrossValidator

logger = logging.getLogger("validation")


def cross_validate(
    data: pd.DataFrame, model: AggregatePredictor, of_cross_validate: dict
):
    """Cross-validate the provided model using the provided data.

    Args:
        data (pandas.DataFrame): The input data.
        model (AggregatePredictor): The model to cross-validate.
        of_cross_validate (dict): The parameters of the cross-validation step
            (typically read from `conf/base/parameters/validate.yaml`).

    Returns:
        CrossValidator: A `CrossValidator` model containing the cross-validation results.
    """
    X_train, y_train = split_training_data(data)
    cv_model = CrossValidator(
        model,
        group_by=of_cross_validate.get("group_by", "week"),
        stratify_by=of_cross_validate.get("stratify_by", "month"),
        n_splits=of_cross_validate.get("n_splits", 3),
        n_repeats=of_cross_validate.get("n_repeats", 5),
        keep_estimators=of_cross_validate.get("keep_estimators", True),
        n_jobs=of_cross_validate.get("n_jobs"),
        verbose=of_cross_validate.get("verbose", True),
        fit_params=of_cross_validate.get("fit_params"),
        random_state=of_cross_validate.get("random_state"),
    )
    cv_model = cv_model.fit(X_train, y_train)
    for metric in cv_model.scorers.keys():
        logger.info(
            f"Mean out-of-sample CV(RMSE) (%): {np.mean(cv_model.scores_[metric])*100}"
        )

    return cv_model


def create_conformal(
    data: pd.DataFrame, cv_model: CrossValidator, of_conformal_predictor: dict
):
    """[summary]

    Args:
        data (pandas.DataFrame): Input data.
        cv_model (CrossValidator): A `CrossValidator` model containing the cross-validation
            results.
        of_conformal_predictor (dict): The parameters of the conformal prediction training step
            (typically read from `conf/base/parameters/validate.yaml`).

    Returns:
        AggregatedCp: The conformal prediction model.
    """
    X_train, y_train = split_training_data(data)
    conformal_model = AggregatedCp(
        estimators=cv_model.estimators,
        oos_masks=cv_model.oos_masks,
        add_normalizer=of_conformal_predictor.get("add_normalizer", True),
        extra_regressors=of_conformal_predictor.get("extra_regressors"),
        n_estimators=of_conformal_predictor.get("n_estimators", 100),
        min_samples_leaf=of_conformal_predictor.get("min_samples_leaf", 0.05),
        max_samples=of_conformal_predictor.get("max_samples", 0.8),
        n_jobs=of_conformal_predictor.get("n_jobs"),
    )
    conformal_model = conformal_model.fit(X_train, y_train)
    return conformal_model
