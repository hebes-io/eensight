# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import pandas as pd

from eensight.models._composite import AggregatePredictor
from eensight.models._conformal import AggregatedCp

from .sampling import generate_samples

logger = logging.getLogger("savings")


def apply_compare(
    data: pd.DataFrame,
    mean_model: AggregatePredictor,
    conformal_model: AggregatedCp,
    of_apply_compare: dict,
):
    """Estimate cumulative energy savings.

    Args:
        data (pandas.DataFrame): The input data
        mean_model (AggregatePredictor): The baseline consumption model.
        conformal_model (AggregatedCp): The conformal prediction model for
            uncertainty estimation.
        of_apply_compare (dict): The parameters of the estimation step (typically
            read from `conf/base/parameters/compare.yaml`).

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    to_drop = data.filter(like="_outlier", axis=1).columns.to_list()
    if len(to_drop) > 0:
        data = data.drop(to_drop, axis=1)

    if "consumption" not in data:
        raise ValueError("Consumption missing from input data")
    y_true = data.pop("consumption")

    significance = np.linspace(0, 1, of_apply_compare.get("steps", 20))[1:-1]

    pred_mean = mean_model.predict(data)
    pred_conf = conformal_model.predict(data, significance)
    samples = generate_samples(
        of_apply_compare.get("n_samples", 200),
        prediction=pred_mean["consumption"],
        quantiles=pred_conf,
    )

    savings = samples.sub(y_true, axis=0)
    keep_n_last = of_apply_compare.get("keep_n_last")
    if keep_n_last is None:
        fract_savings = savings.apply(np.cumsum, axis=0).div(
            pred_mean["consumption"].cumsum(), axis=0
        )
    elif keep_n_last == 1:
        fract_savings = pd.DataFrame.from_dict(
            {
                savings.index[-1]: savings.sum(axis=0)
                .div(pred_mean["consumption"].sum(), axis=0)
                .to_dict()
            },
            orient="index",
        )
    else:
        fract_savings = (
            savings.apply(np.cumsum, axis=0)
            .div(pred_mean["consumption"].cumsum(), axis=0)
            .iloc[-keep_n_last:]
        )

    logger.info(f"Mean of fractional savings (%) {100 * fract_savings.iloc[-1].mean()}")
    logger.info(
        f"Standard deviation of fractional savings (%) {100 * fract_savings.iloc[-1].std()}"
    )
    return fract_savings
