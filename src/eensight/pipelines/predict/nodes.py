# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging

import pandas as pd
from feature_encoders.utils import as_series

from eensight.metrics import mpiw, picp
from eensight.models._composite import AggregatePredictor
from eensight.models._conformal import AggregatedCp

logger = logging.getLogger("prediction")


def apply_predict(
    data: pd.DataFrame,
    mean_model: AggregatePredictor,
    conformal_model: AggregatedCp,
    of_apply_predict: dict,
):
    """Predict using the input data.

    Args:
        data (pandas.DataFrame): The input data
        mean_model (AggregatePredictor): The baseline consumption model.
        conformal_model (AggregatedCp): The conformal prediction model for
            uncertainty estimation.
        of_apply_predict (dict): The parameters of the prediction step (typically
            read from `conf/base/parameters/predict.yaml`).

    Returns:
        pandas.DataFrame: Dataframe with mean prediction and lower and upper intervals.
    """
    to_drop = data.filter(like="_outlier", axis=1).columns.to_list()
    if len(to_drop) > 0:
        data = data.drop(to_drop, axis=1)

    y_true = None
    if "consumption" in data:
        y_true = data.pop("consumption")

    pred_mean = mean_model.predict(data)
    pred_conf = conformal_model.predict(
        data, of_apply_predict.get("significance", 0.90)
    )

    prediction = pd.concat(
        (
            pred_mean[["consumption"]],
            (pred_mean["consumption"] + as_series(pred_conf)).to_frame(
                "consumption_high"
            ),
            (pred_mean["consumption"] - as_series(pred_conf)).to_frame(
                "consumption_low"
            ),
        ),
        axis=1,
    )

    logger.info(f"Significance level: {of_apply_predict.get('significance', 0.90)}")
    if y_true is not None:
        logger.info(
            "Prediction Interval Coverage Probability (PICP): "
            f"{picp(y_true, prediction['consumption_low'], prediction['consumption_high'])}"
        )
    logger.info(
        "Mean Prediction Interval Width (MPIW): "
        f"{mpiw(prediction['consumption_low'], prediction['consumption_high'])}"
    )

    return prediction
