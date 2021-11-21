# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from feature_encoders.generate import DatetimeFeatures
from feature_encoders.utils import as_series, get_categorical_cols
from scipy import stats

from eensight.models import BoostedTreeRegressor


def get_resid(
    data: pd.DataFrame, period: Union[List, Tuple, int], cat_columns: List[str]
):
    """Fit a predictive model for energy consumption and return residuals.

    Args:
        data (pandas.DataFrame): The input dataframe.
        period (Union[List, Tuple, int]): The year or list of years for the
            data to use.
        cat_columns (List[str]): A list with the names of the columns contating
            categorical data.

    Returns:
        pandas.Series: The residuals of the predictive model.
    """
    X = data.drop("consumption", axis=1)
    y = data["consumption"]

    if isinstance(period, (list, tuple)):
        X = X[X.index.year.isin(period)]
    else:
        X = X[X.index.year == period]
    y = y.loc[X.index]

    cat_features = None
    if cat_columns:
        cat_features = [X.columns.get_loc(c) for c in cat_columns]

    model = BoostedTreeRegressor(cat_features=cat_features, iterations=500)
    model = model.fit(X, y)
    pred = model.predict(X)
    return y - as_series(pred)


def apply_paired_learners(
    data: pd.DataFrame,
    long_period: Union[List, Tuple],
    short_period: int,
    cat_columns: List[str] = None,
    stat_size: int = 150,
    stat_distance: float = 0.1,
    window: int = 300,
    alpha: float = 0.005,
):
    """Apply a pair of learners to detect data drift.

    Args:
        data (pandas.DataFrame): The input dataframe.
        long_period (Union[List, Tuple]): The years over which the first learner will
            be trained.
        short_period (int): The year over which the second learner will be trained.
        cat_columns (List[str], optional): A list with the names of the columns contating
            categorical data. Defaults to None.
        stat_size (int, optional): Number of non-NaN hours in the sliding window for a
            valid test. Defaults to 100.
        stat_distance (float, optional): The minimum distance between the distributions of
            the two samples of residuals to test for statistical difference. Defaults to 0.3.
        window (int, optional): Size of the sliding window in hours. Defaults to 300.
        alpha (float, optional): Probability for the test statistic of the Kolmogorov-Smirnov
            test. The alpha parameter is very sensitive, therefore should be set below 0.01.
            Defaults to 0.001.

    Returns:
        pandas.Series: A Series where the value is 1 if drift is detected and 0 otherwise.
    """
    data = data[~data["drift"]]
    resid_long = get_resid(data, long_period, cat_columns)
    resid_short = get_resid(data, short_period, cat_columns)
    resid_long = resid_long[resid_short.index]

    def sample_test(x):
        right = resid_short[x.index].dropna()
        x = x.dropna()

        if (len(x) < stat_size) or (len(right) < stat_size):
            return False
        else:
            (st, p_value) = stats.ks_2samp(x, right, mode="exact")
            if p_value <= alpha and st > stat_distance:
                return True
            else:
                return False

    forward = (
        resid_long.rolling(window, center=False).apply(sample_test).fillna(value=0)
    )
    backward = (
        resid_long[::-1]
        .rolling(window, center=False)
        .apply(sample_test)
        .fillna(value=0)
    )
    return np.logical_or(forward, backward)


def detect_drift(
    data: pd.DataFrame,
    stat_size: int = 150,
    stat_distance: float = 0.1,
    window: int = 300,
    alpha: float = 0.005,
    logger: logging.Logger = None,
):
    """Detect whether the given data exhibits data drift.

    Args:
        data (pandas.DataFrame): The input dataframe.
        stat_size (int, optional): Number of non-NaN hours in the sliding window for a
            valid test. Defaults to 150.
        stat_distance (float, optional): The minimum distance between the distributions of
            the two samples of residuals to test for statistical difference. Defaults to 0.3.
        window (int, optional): Size of the sliding window in hours. Defaults to 300.
        alpha (float, optional): Probability for the test statistic of the Kolmogorov-Smirnov
            test. The alpha parameter is very sensitive, therefore should be set below 0.01.
            Defaults to 0.001.
        logger (logging.Logger, optional): A logger to use. If None, a new logger will be
            created. Defaults to None.

    Returns:
        pandas.DataFrame: The input data with a new column `drift` indicating where drift was
            detected.
    """
    if logger is None:
        logger = logging.getLogger("drift-detection")

    dates = data.index.to_series()
    dt = dates.diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()
    steps_per_hour = math.ceil(pd.Timedelta("1H") / time_step)
    window *= steps_per_hour
    stat_size *= steps_per_hour

    years = data.groupby(lambda x: x.year).first().index.to_list()
    if len(years) > 1:
        cat_columns = get_categorical_cols(data, int_is_categorical=False)
        data_ext = (
            data.interpolate(method="slinear")
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        data_ext = DatetimeFeatures(
            subset=["month", "dayofweek", "hour"]
        ).fit_transform(data_ext)
        data_ext["drift"] = False

        years.reverse()
        for i in range(1, len(years)):
            logger.info(
                f"Applying paired learners, long period: {years[: i + 1]}, short period: {years[i]}"
            )
            drift = apply_paired_learners(
                data_ext,
                years[: i + 1],
                years[i],
                cat_columns=cat_columns,
                stat_size=stat_size,
                stat_distance=stat_distance,
                window=window,
                alpha=alpha,
            )
            data_ext.loc[drift[drift == 1].index, "drift"] = True
            data_ext.loc[drift[drift.isna()].index, "drift"] = True
    else:
        data_ext = pd.DataFrame(False, index=data.index, columns=["drift"])

    return pd.concat((data, data_ext[["drift"]]), axis=1)
