# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import calendar

import pandas as pd
from feature_encoders.models import SeasonalPredictor
from sklearn.utils import Bunch


def decompose_consumption(
    data: pd.DataFrame,
    ds: str = None,
    add_trend: bool = False,
    min_samples: float = 0.8,
    alpha: float = 0.01,
    return_model: bool = False,
):
    """Apply seasonal decomposition on energy consumption data.

    Args:
        data (pd.DataFrame): Input dataframe.
        ds (str, optional): The name of the input dataframe's column that
            contains datetime information. If None, it is assumed that the
            datetime information is provided by the input dataframe's index.
            Defaults to None.
        add_trend (bool, optional): If True, a linear time trend will be
            added. Defaults to False.
        min_samples (float ([0, 1]), optional): Minimum number of samples
            chosen randomly from original data by the RANSAC (RANdom SAmple
            Consensus) algorithm. Defaults to 0.8.
        alpha (float, optional): Parameter for the underlying ridge estimator
            (`base_estimator`). It must be a positive float. Regularization
            improves the conditioning of the problem and reduces the variance
            of the estimates. Larger values specify stronger regularization.
            Defaults to 0.01.
        return_model (bool, optional): Whether to return the fitted model.
            Defaults to False.

    Returns:
        sklearn.utils.Bunch: Dict-like structure the prediction and fitted model.
    """
    model = SeasonalPredictor(
        ds=ds,
        add_trend=add_trend,
        yearly_seasonality="auto",
        weekly_seasonality=False,
        daily_seasonality=False,
        min_samples=min_samples,
        alpha=alpha,
    )

    X = data.drop("consumption", axis=1)
    y = data["consumption"]

    dates = X.index.to_series() if ds is None else X[ds]
    X["dayofweek"] = dates.dt.dayofweek.map(lambda x: calendar.day_abbr[x])
    X = X.merge(pd.get_dummies(X["dayofweek"]), left_index=True, right_index=True).drop(
        "dayofweek", axis=1
    )

    for i in range(7):
        day = calendar.day_abbr[i]
        model.add_seasonality(
            f"daily_on_{day}", period=1, fourier_order=4, condition_name=day
        )

    model = model.fit(X, y)
    prediction = model.predict(X)
    return Bunch(
        prediction=prediction,
        model=None if not return_model else model,
    )


def decompose_temperature(
    data: pd.DataFrame,
    ds: str = None,
    min_samples: float = 0.8,
    alpha: float = 0.01,
    return_model: bool = False,
):
    """Apply seasonal decomposition on outdoor air temperature data.

    Args:
        data (pd.DataFrame): Input dataframe.
        ds (str, optional): The name of the input dataframe's column that
            contains datetime information. If None, it is assumed that the
            datetime information is provided by the input dataframe's index.
            Defaults to None.
        min_samples (float ([0, 1]), optional): Minimum number of samples
            chosen randomly from original data by the RANSAC (RANdom SAmple
            Consensus) algorithm. Defaults to 0.8.
        alpha (float, optional): Parameter for the underlying ridge estimator
            (`base_estimator`). It must be a positive float. Regularization
            improves the conditioning of the problem and reduces the variance
            of the estimates. Larger values specify stronger regularization.
            Defaults to 0.01.
        return_model (bool, optional): Whether to return the fitted model.
            Defaults to False.

    Returns:
        sklearn.utils.Bunch: Dict-like structure the prediction and fitted model.
    """
    model = SeasonalPredictor(
        ds=ds,
        add_trend=False,
        yearly_seasonality="auto",
        weekly_seasonality=False,
        daily_seasonality="auto",
        min_samples=min_samples,
        alpha=alpha,
    )

    X = data.drop("temperature", axis=1)
    y = data["temperature"]
    model = model.fit(X, y)
    prediction = model.predict(X)
    return Bunch(
        prediction=prediction,
        model=None if not return_model else model,
    )
