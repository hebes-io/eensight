# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Union

import pandas as pd

from eensight.methods.prediction.activity import adjust_activity
from eensight.methods.prediction.baseline import UsagePredictor

logger = logging.getLogger("eensight")


def adjust_activity_levels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model: UsagePredictor,
    non_occ_features: Union[str, List[str]],
    cat_features: Union[str, List[str]],
    assume_hurdle: bool,
    n_trials: int,
    upper_bound: float,
    verbose: bool,
):
    """Adjust activity levels for events that affect density of energy consumption.

    Args:
        features (pd.DataFrame): The feature data.
        labels (pd.DataFrame): The target data.
        model (UsagePredictor): A trained model that predicts energy consumption given
            `non_occ_features` and activity levels.
        non_occ_features (str or list of str): The names of the occupancy-independent
            features to use for estimating activity levels.
        cat_features (str or list of str): The name(s) of the categorical features in
            the input data.
        assume_hurdle (bool): If True, it is assumed that the energy consumption data is
            generated from a hurdle model.
        n_trials (int): The number of iterations for the underlying optimization.
        upper_bound (float): The maximum value that the true upper bound for the adjusted
            activity levels can take.
        verbose (bool): If True, a progress bar is visible.

    Returns:
        pandas.DataFrame: The adjusted activity time series.
    """

    activity, best_params = adjust_activity(
        features,
        labels,
        model,
        non_occ_features=non_occ_features,
        cat_features=cat_features,
        assume_hurdle=assume_hurdle,
        n_trials=n_trials,
        upper_bound=upper_bound,
        verbose=verbose,
        return_params=True,
    )

    logger.info(f"Estimated parameters: {best_params}")
    return activity.to_frame("activity")
