# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from datetime import date, time

import numpy as np
import pandas as pd

from .metric_learning import learn_distance_metric
from .prototypes import find_prototypes, get_matrix_profile

logger = logging.getLogger("day-typing-stage")


def prepare_data(data, prepare_daytyping):
    data = data[["consumption"]].mask(data["consumption_outlier"], np.nan)
    remove = data["consumption"].isna().groupby(
        lambda x: x.date()
    ).sum() > prepare_daytyping.get("nan_threshold", 0.5)
    data = data[~np.isin(data.index.date, remove[remove].index)]
    data["consumption"] = data["consumption"].interpolate(method="nearest")
    return data


def get_daily_matrix_profile(data):
    profile = get_matrix_profile(data["consumption"], window=24)
    return profile[profile.index.time == time(0, 0)]


def apply_metric_learning(
    data, profile, start_time, end_time, for_find_prototypes, for_distance_metric
):
    prototype_results = find_prototypes(
        data,
        profile,
        start_time,
        end_time=end_time,
        max_iter=for_find_prototypes.get("max_iter", 30),
        early_stopping=True,
        early_stopping_val=for_find_prototypes.get("early_stopping_val", 0.1),
    )

    prototypes = data.iloc[prototype_results["prototypes"]].index
    distances = prototype_results.distance_from_prototypes
    metric_results = learn_distance_metric(
        distances,
        pairs_per_prototype=for_distance_metric.get("pairs_per_prototype", 50),
        test_size=for_distance_metric.get("test_size", 0.25),
    )
    logger.info(f"Distance metric learning score: {metric_results.score}")
    return prototypes, metric_results.metric_components


def create_time_intervals(window):
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


def apply_day_typing(data, daytyping_window, for_find_prototypes, for_distance_metric):
    X = data["consumption"]
    window = int(daytyping_window)

    profile = get_matrix_profile(X, window=window)
    intervals = create_time_intervals(window)

    prototype_days = {}
    distance_metrics = {}

    for (start_time, end_time) in intervals:
        prototypes, metric_components = apply_metric_learning(
            X, profile, start_time, end_time, for_find_prototypes, for_distance_metric
        )

        prototype_days[start_time] = prototypes
        distance_metrics[start_time] = {
            "end_time": end_time,
            "metric_components": metric_components,
        }
    return prototype_days, distance_metrics
