# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import datetime

import numpy as np
import pandas as pd


def merge_hours_hours(left: pd.DataFrame, right: pd.DataFrame):
    """Merge two dataframes that both have a time step <=pd.Timedelta("1H").

    Args:
        left (pandas.DataFrame): The first dataframe to merge.
        right (pandas.DataFrame): The second dataframe to merge.

    Returns:
        pandas.DataFrame: The merged dataframe.
    """
    merged = pd.merge_asof(
        left,
        right,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("1H"),
    )
    return merged


def merge_hours_days(left: pd.DataFrame, right: pd.DataFrame):
    """Merge two dataframes where the first has a time step <=pd.Timedelta("1H")
    and the second has a time step ==pd.Timedelta("1D").

    Args:
        left (pd.DataFrame): The first dataframe to merge (hourly resolution).
        right (pd.DataFrame): The second dataframe to merge (daily resolution).

    Returns:
        pandas.DataFrame: The merged dataframe.
    """
    if isinstance(right.index[0], datetime.date):
        right.index = right.index.map(pd.to_datetime)

    start = left.index.time.min()
    right.index = right.index.map(pd.to_datetime).map(
        lambda x: x.replace(hour=start.hour, minute=start.minute, second=start.second)
    )

    right_to_hourly = left.index.to_frame().join(right).loc[:, right.columns]
    for _, grouped in right_to_hourly.groupby(lambda x: x.date):
        if np.any(grouped.iloc[[0]].notna()):
            right_to_hourly.loc[grouped.index, :] = right_to_hourly.loc[
                grouped.index, :
            ].fillna(method="ffill")

    return merge_hours_hours(left, right_to_hourly)


def recover_missing_dates(data: pd.DataFrame):
    """Add missing timestamps.

    If there are missing timestamps, they are added and the respective data is treated
    as missing values.

    Args:
        data (pandas.DataFrame): The data to correct.

    Returns:
        pandas.DataFrame: The input data with the missing timestamps added.
    """
    dt = data.index.to_series().diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()

    if time_step == pd.Timedelta("0 days 00:00:00"):
        raise ValueError("Input data contains dublicate dates")

    full_range = pd.date_range(
        start=datetime.datetime.combine(data.index.min().date(), datetime.time(0, 0)),
        end=datetime.datetime.combine(
            data.index.max().date() + datetime.timedelta(days=1), datetime.time(0, 0)
        ),
        freq=time_step,
    )[:-1]

    index_name = data.index.name
    data = pd.DataFrame(index=full_range).join(data, how="left")
    data.index.set_names(index_name, inplace=True)
    return data


def create_groups(X: pd.DataFrame, group_block: str):
    """Greate groups for the input data.

    Args:
        X (pandas.DataFrame): The input data.
        group_block ({'day', 'week', 'month'}): Parameter that defines what
            constitutes an indivisible group of data.

    Raises:
        ValueError: If `group_block` is not one of 'day', 'week' or 'month'.

    Returns:
        pandas Series: The groups for all observations in input data.
    """
    if group_block == "day":
        grouped = X.groupby(lambda x: x.dayofyear)
    elif group_block == "week":
        grouped = X.groupby(lambda x: x.isocalendar()[1])
    elif group_block == "month":
        grouped = X.groupby(lambda x: x.month)
    else:
        raise ValueError("`groups` can be either `day`, `week` or `month`.")

    groups = None
    for i, (_, group) in enumerate(grouped):
        groups = pd.concat([groups, pd.Series(i, index=group.index)])
    return groups.reindex(X.index)


def split_training_data(data):
    # We train the predictive models without outliers:
    X_train = data.loc[~data["consumption_outlier"]].drop(
        ["consumption", "consumption_outlier"], axis=1
    )
    y_train = data.loc[X_train.index, ["consumption"]]
    return X_train, y_train
