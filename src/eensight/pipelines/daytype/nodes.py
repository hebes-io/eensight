# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
from datetime import date, time

import numpy as np
import pandas as pd

from .metric_learning import learn_distance_metric
from .prototypes import find_prototypes, get_matrix_profile

logger = logging.getLogger("day-typing")


def prepare_data(data: pd.DataFrame, of_prepare_data: dict):
    """Prepare the data for the subsequent nodes.

    Args:
        data (pandas.DataFrame): The input data.
        of_prepare_data (dict): The parameters of the preparation step (typically
            read from `conf/base/parameters/daytype.yaml`).

    Returns:
        pandas.DataFrame: The data for the subsequent nodes.
    """
    data = data[["consumption"]].mask(data["consumption_outlier"], np.nan)
    remove = data["consumption"].isna().groupby(
        lambda x: x.date()
    ).sum() > of_prepare_data.get("nan_threshold", 0.33)
    data = data[~np.isin(data.index.date, remove[remove].index)]
    data["consumption"] = (
        data["consumption"]
        .interpolate(method="slinear")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )
    return data


def get_daily_matrix_profile(data: pd.DataFrame):
    """Calculate the daily matrix profiles for the input data.

    Args:
        data (pandas.DataFrame): The input data.

    Returns:
        pandas.DataFrame: The matrix profile values.
    """
    profile = get_matrix_profile(data["consumption"], window=24)
    return profile[profile.index.time == time(0, 0)]


def _apply_metric_learning(
    data, profile, start_time, end_time, of_find_prototypes, of_distance_learning
):
    prototype_results = find_prototypes(
        data,
        profile,
        start_time,
        end_time=end_time,
        max_iter=of_find_prototypes.get("max_iter", 30),
    )

    stopping_metric = pd.Series(prototype_results.mmd_scores)
    selected = (
        (stopping_metric.abs() < of_find_prototypes.get("early_stopping_val", 0.1))
        .rolling(2)
        .sum()
        == 2
    ).idxmax() + 1

    distances = prototype_results.distance_from_prototypes.iloc[:, :selected]
    metric_results = learn_distance_metric(
        distances,
        pairs_per_prototype=int(
            len(np.unique(data.index.date))
            * of_distance_learning.get("pairs_per_prototype", 0.15)
        ),
        test_size=of_distance_learning.get("test_size", 0.25),
    )
    logger.info(f"Distance metric learning score: {metric_results.score}")
    return (
        data.iloc[prototype_results.prototypes].index,
        metric_results.metric_components,
    )


def _create_time_intervals(window):
    intervals = pd.date_range(
        start=date(2000, 1, 1), end=date(2000, 1, 2), freq=f"{window}H", closed="left"
    )

    if len(intervals) == 1:
        intervals = [(time(0, 0), None)]
    else:
        last = intervals[-1]
        intervals = [
            (i.time(), j.time()) for i, j in zip(intervals[:-1], intervals[1:])
        ]
        intervals.append((last.time(), None))

    return intervals


def apply_day_typing(
    data: pd.DataFrame,
    daytyping_window: int,
    of_find_prototypes: dict,
    of_distance_learning: dict,
):
    """Carry out the day typing step.

    Args:
        data (pandas.DataFrame): The input data.
        daytyping_window (int): The number of hours over which the method should search
            for prototype profiles.
        of_find_prototypes (dict): The parameters of the prototype identification step
            (typically read from `conf/base/parameters/daytype.yaml`).
        of_distance_learning (dict): The parameters of the distance learning step
            (typically read from `conf/base/parameters/daytype.yaml`).

    Returns:
        tuple containing

        - **prototype_days** (*pandas.DatetimeIndex*): The prototype days.
        - **distance_metrics** (*dict*): A dictionary that contains time interval
            information of the form: key: interval number, values: interval start
            time, interval end time, and components of the corresponding distance
            metric.
    """
    X = data["consumption"]
    window = int(daytyping_window)

    profile = get_matrix_profile(X, window=window)
    intervals = _create_time_intervals(window)

    prototype_days = {}
    distance_metrics = {}

    for i, (start_time, end_time) in enumerate(intervals):
        prototypes, metric_components = _apply_metric_learning(
            X, profile, start_time, end_time, of_find_prototypes, of_distance_learning
        )

        prototype_days[i] = prototypes
        distance_metrics[i] = {
            "start_time": start_time,
            "end_time": end_time,
            "metric_components": metric_components,
        }
    return prototype_days, distance_metrics
