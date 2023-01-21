# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("eensight")


def expand_dates(data: pd.DataFrame) -> pd.DataFrame:
    """Add missing timestamps.

    If there are missing timestamps, they are added and the respective data is treated
    as missing values.

    Args:
        data (pandas.DataFrame): The data to correct. The dataframe's index must be a
            pandas DatetimeIndex and in increasing order.

    Returns:
        pandas.DataFrame: The input data with the missing timestamps added.
    """

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The input dataframe's index must be a pandas DatetimeIndex.")

    if not data.index.is_monotonic_increasing:
        raise ValueError("The input dataframe's index must be in increasing order.")

    dt = data.index.to_series().diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()

    if time_step >= pd.Timedelta(days=1):
        return data

    if time_step == pd.Timedelta("0 days 00:00:00"):
        raise ValueError("Input data contains dublicate dates")

    full_index = pd.date_range(
        start=datetime.datetime.combine(data.index.min().date(), datetime.time(0, 0)),
        end=datetime.datetime.combine(
            data.index.max().date() + datetime.timedelta(days=1), datetime.time(0, 0)
        ),
        freq=time_step,
        inclusive="left",
    )
    return data.reindex(full_index)


def filter_data(
    X: pd.Series,
    *,
    min_value: float = None,
    max_value: float = None,
    allow_zero: bool = True,
    allow_negative: bool = False,
    copy=True,
) -> pd.Series:
    """Screen for non-physically plausible values in the data.

    Args:
        X (pandas Series): Time series data to filter.
        min_value (float, optional): Minumun acceptable value. Values lower than
            `min_value` will be replaced by NaN. Defaults to None.
        max_value (float, optional): Maximum acceptable value. Values greater than
            `max_value` will be replaced by NaN. Defaults to None.
        allow_zero (bool, optional): Whether zero values should be allowed. Defaults
            to True.
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
        raise ValueError("Input data is expected of pandas Series")

    if copy:
        X = X.copy()

    nan_before = X.isna().sum()

    if min_value is not None:
        X.loc[X < min_value] = np.nan
    if max_value is not None:
        X.loc[X > max_value] = np.nan
    if not allow_zero:
        X.loc[X.abs() <= 1e-05] = np.nan
    if not allow_negative:
        X.loc[X < 0] = np.nan

    nan_after = X.isna().sum()
    if nan_after > nan_before:
        logger.info(f"{nan_after - nan_before} NaN values added by filtering.")

    return X


def check_data_adequacy(
    features: pd.DataFrame, labels: pd.DataFrame, max_missing_pct: float = 0.1
) -> pd.DataFrame:
    """Check whether enough data is available for each month of the year.

    Args:
        features (pandas.DataFrame of shape (n_samples, n_features)): The features dataframe.
        labels (pandas.DataFrame of shape (n_samples, 1)): The labels dataframe.
        max_missing_pct (float, optional): The maximum acceptable percentage of missing
            data per month. Defaults to 0.1.

    Raises:
        ValueError: If `max_missing_pct` is larger than 0.99 or less than 0.

    Returns:
        pandas.DataFrame: Dataframe containing percentage of missing data per month.
    """
    if (max_missing_pct > 0.99) or (max_missing_pct < 0):
        raise ValueError(
            "Values for `max_missing_pct` cannot be larger than "
            "0.99 or smaller than 0"
        )

    missing = labels.mask(features.isna().all(axis=1), np.nan)
    missing_per_month = dict()

    for month, group in missing.groupby(lambda x: x.month):
        missing_per_month[month] = (
            np.sum(
                group.groupby([lambda x: x.day, lambda x: x.hour]).count() == 0
            ).item()
            / 720  # hours per month
        )

    missing_per_month = {f"M{key}": val for key, val in missing_per_month.items()}
    missing_per_month = pd.DataFrame.from_dict(
        missing_per_month, orient="index", columns=["missing_pct"]
    )

    insufficient = missing_per_month[missing_per_month["missing_pct"] > max_missing_pct]

    if len(insufficient) > 0:
        logger.warning(
            f"Months with not enough data are:\n {insufficient['missing_pct'].to_dict()}"
        )
    return missing_per_month
