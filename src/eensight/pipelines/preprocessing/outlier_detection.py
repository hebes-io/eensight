# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import pandas as pd


def global_filter(
    X: pd.Series,
    no_change_window: int = 3,
    min_value: float = None,
    max_value: float = None,
    allow_zero: bool = False,
    allow_negative: bool = False,
    copy=True,
) -> pd.Series:

    if not isinstance(X, pd.Series):
        raise ValueError("Input data is expected of pd.Series type")

    if copy:
        X = X.copy()

    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta("1H") / time_step)
    start = int(no_change_window * steps_per_hour)

    changes = X.diff().abs()
    X[start:] = X[start:].mask(
        changes.rolling(f"{no_change_window}H").sum() < 1e-3, np.nan
    )

    if min_value is not None:
        X.loc[X < min_value] = np.nan
    if max_value is not None:
        X.loc[X > max_value] = np.nan
    if not allow_zero:
        X.loc[X.abs() <= np.finfo(np.float32).eps] = np.nan
    if not allow_negative:
        X.loc[X < 0] = np.nan

    median = X.median()
    X.loc[X.abs() > 10 * median] = np.nan
    return X


def global_outlier_detect(X: pd.Series, c: float = 5) -> pd.Series:
    X = X / X.median()
    scale = X.mad()

    outlier_score = np.maximum(np.abs(X - 1) - c * scale, 0)
    outlier_score = outlier_score.fillna(value=0)
    return outlier_score


def local_outlier_detect(
    X: pd.Series, min_samples: float = 0.6, c: float = 5
) -> pd.Series:
    dates = X.index.to_series()
    dt = dates.diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()
    steps_per_hour = math.ceil(pd.Timedelta("1H") / time_step)
    min_samples = int(24 * steps_per_hour * min_samples)

    median_daily = X.groupby(lambda x: x.date).median().to_dict()
    median_daily = dates.map(lambda x: median_daily[x.date()])
    X = X / median_daily

    mad_daily = X.groupby(lambda x: x.date).mad().to_dict()
    mad_daily = dates.map(lambda x: mad_daily[x.date()])
    count_daily = X.groupby(lambda x: x.date).count().to_dict()
    count_daily = dates.map(lambda x: count_daily[x.date()])

    outlier_score = np.maximum(np.abs(X - 1) - c * mad_daily, 0)
    outlier_score = outlier_score.mask(count_daily < min_samples, 0)
    return outlier_score
