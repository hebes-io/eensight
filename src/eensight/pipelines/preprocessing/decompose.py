# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import calendar

import pandas as pd
from sklearn.utils import Bunch

from eensight.models.seasonal import SeasonalDecomposer


def decompose_consumption(
    X: pd.DataFrame,
    feature: str = "consumption",
    dt: str = None,
    add_trend: bool = False,
    alpha=1,
    return_conditions=False,
    return_model=False,
):
    model = SeasonalDecomposer(
        feature,
        dt=dt,
        add_trend=add_trend,
        yearly_seasonality="auto",
        weekly_seasonality=False,
        daily_seasonality=False,
        alpha=alpha,
    )

    dates = X.index.to_series() if dt is None else X[dt]
    columns_before = X.columns
    X = X[[feature]].copy()

    X["dayofweek"] = dates.dt.dayofweek.map(lambda x: calendar.day_abbr[x])
    X = X.merge(pd.get_dummies(X["dayofweek"]), left_index=True, right_index=True).drop(
        "dayofweek", axis=1
    )

    for i in range(7):
        day = calendar.day_abbr[i]
        model.add_seasonality(
            f"daily_on_{day}", period=1, fourier_order=4, condition_name=day
        )

    transformed = model.fit_transform(X)
    conditions = set(X.columns) - set(columns_before)
    return Bunch(
        transformed=transformed,
        conditions=None if not return_conditions else X[conditions],
        model=None if not return_model else model,
    )


def decompose_temperature(
    X: pd.DataFrame,
    feature: str = "temperature",
    dt: str = None,
    alpha=1,
    return_model=False,
):
    model = SeasonalDecomposer(
        feature,
        dt=dt,
        add_trend=False,
        yearly_seasonality="auto",
        weekly_seasonality=False,
        daily_seasonality="auto",
        alpha=alpha,
    )

    transformed = model.fit_transform(X)
    return Bunch(
        transformed=transformed,
        conditions=None,
        model=None if not return_model else model,
    )
