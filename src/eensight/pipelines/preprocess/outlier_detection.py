# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

logger = logging.getLogger("preprocess")


def global_filter(
    X: pd.Series,
    no_change_window: int = 3,
    max_pct_of_dummy: float = 0.05,
    min_value: float = None,
    max_value: float = None,
    allow_zero: bool = False,
    allow_negative: bool = False,
    copy=True,
) -> pd.Series:
    """Screen for non-physically plausible values in the data.

    Args:
        X (pd.Series): Input data.
        no_change_window (int, optional): Streaks of constant values that
            last longer that `no_change_window` will be replaced by NaN.
            Defaults to 3.
        max_pct_of_dummy (float, optional): Long streaks of constant values
            will not be filtered out if they represent more than `max_pct_of_dummy`
            percentange of the dataset. Defaults to 0.05.
        min_value (float, optional): Minumun acceptable value. Values lower than
            `min_value` will be replaced by NaN. Defaults to None.
        max_value (float, optional): Maximum acceptable value. Values greater than
            `max_value` will be replaced by NaN. Defaults to None.
        allow_zero (bool, optional): Whether zero values should be allowed. Defaults
            to False.
        allow_negative (bool, optional): Whether negative values should be allowed.
            Defaults to False.
        copy (bool, optional): Whether the NaN replacements should performed on a copy
            of the input data. Defaults to True.

    Raises:
        ValueError: If input data is not pandas Series.

    Returns:
        pandas.Series: The filtered data.
    """
    if not isinstance(X, pd.Series):
        raise ValueError("Input data is expected of pd.Series type")

    if copy:
        X = X.copy()

    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta("1H") / time_step)
    start = int(no_change_window * steps_per_hour)

    changes = X.diff().abs()
    seems_dummy_data = changes.rolling(f"{no_change_window}H").sum() < 1e-05
    if seems_dummy_data.sum() <= max_pct_of_dummy * len(X):
        X[start:] = X[start:].mask(
            changes.rolling(f"{no_change_window}H").sum() < 1e-05, np.nan
        )
    else:
        logger.warning(
            "Large number of constant values found. Replacing them with NaN was skipped"
        )

    if min_value is not None:
        X.loc[X < min_value] = np.nan
    if max_value is not None:
        X.loc[X > max_value] = np.nan
    if not allow_zero:
        X.loc[X.abs() <= np.finfo(np.float32).eps] = np.nan
    if not allow_negative:
        X.loc[X < 0] = np.nan

    return X


def global_outlier_detect(X: pd.Series, c: float = 7) -> pd.Series:
    """Identify potential outliers by comparing each value to the overall
    median of all values.

    Args:
        X (pd.Series): Input data.
        c (float, optional): Factor used to determine the bound of normal
            range (between median_all-c*mad_all and median_all+c*mad_all,
            where median_all is the median of all values and mad_all is
            the median absolute deviation of all values in the input data).
            Defaults to 7.

    Raises:
        ValueError: If input data is not pandas Series.

    Returns:
        pandas.Series: outlier scores. Zero values mean no outlier, non zero
            values mean probable outlier, and the higher the score, the higher
            the probability.
    """
    if not isinstance(X, pd.Series):
        raise ValueError("Input data is expected of pd.Series type")

    X = X / X.median()
    scale = median_abs_deviation(X, scale=1, nan_policy="omit")

    outlier_score = np.maximum(np.abs(X - 1) - c * scale, 0)
    outlier_score = outlier_score.fillna(value=0)
    return outlier_score


def local_outlier_detect(
    X: pd.Series, min_samples: float = 0.66, c: float = 7
) -> pd.Series:
    """Identify potential outliers by comparing each value to the median
    of all values for the corresponding date.

    Args:
        X (pd.Series): Input data.
        min_samples (float, optional): The minimum percentage of observations
            that must be available for any given day so that to take the daily
            statistics into account. If the number of the available observations
            is lower than this threshold, the outlier score is zero. Defaults to
            0.66.
        c (float, optional): Factor used to determine the bound of normal
            range (between median_day-c*mad_day and median_day+c*mad_day,
            where median_day is the median of all values in the observation's
            corresponding day and mad_day is the median absolute deviation of
            all values in the corresponding day). Defaults to 7.

    Raises:
        ValueError: If input data is not pandas Series.

    Returns:
        pandas.Series: outlier scores. Zero values mean no outlier, non zero
            values mean probable outlier, and the higher the score, the higher
            the probability.
    """
    if not isinstance(X, pd.Series):
        raise ValueError("Input data is expected of pd.Series type")

    dates = X.index.to_series()
    dt = dates.diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()
    steps_per_hour = math.ceil(pd.Timedelta("1H") / time_step)
    min_samples = int(24 * steps_per_hour * min_samples)

    median_daily = X.groupby(lambda x: x.date).median().to_dict()
    median_daily = dates.map(lambda x: median_daily[x.date()])
    X = X / median_daily

    mad_daily = (
        X.groupby(lambda x: x.date)
        .apply(lambda x: median_abs_deviation(x, scale=1, nan_policy="omit"))
        .to_dict()
    )
    mad_daily = dates.map(lambda x: mad_daily[x.date()])
    count_daily = X.groupby(lambda x: x.date).count().to_dict()
    count_daily = dates.map(lambda x: count_daily[x.date()])

    outlier_score = np.maximum(np.abs(X - 1) - c * mad_daily, 0)
    outlier_score = outlier_score.mask(count_daily < min_samples, 0)
    return outlier_score
