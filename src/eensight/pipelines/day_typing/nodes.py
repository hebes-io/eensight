# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from datetime import date, time

import numpy as np
import pandas as pd
from joblib import Parallel
from sklearn.utils.fixes import delayed

from .metric_learning import learn_distance_metric
from .prototypes import find_prototypes, get_matrix_profile

logger = logging.getLogger("day-typing-stage")


def prepare_data(data, parameters):
    params = parameters["prepare_for_daytyping"]
    data = data[["consumption"]].mask(data["consumption_outlier"], np.nan)
    remove = data["consumption"].isna().groupby(lambda x: x.date()).sum() > params.get(
        "nan_threshold", 0.5
    )
    data = data[~np.isin(data.index.date, remove[remove].index)]
    data["consumption"] = data["consumption"].interpolate(method="nearest")
    return data


def get_daily_matrix_profile(data):
    profile = get_matrix_profile(data["consumption"], window=24)
    return profile[profile.index.time == time(0, 0)]


def apply_metric_learning(data, profile, start_time, end_time, parameters):
    params = parameters["find_prototypes"]
    prototype_results = find_prototypes(
        data,
        profile,
        start_time,
        end_time=end_time,
        max_iter=params.get("max_iter", 30),
        early_stopping=True,
        early_stopping_val=params.get("early_stopping_val", 0.1),
    )

    params = parameters["learn_distance_metric"]
    distances = prototype_results.distance_from_prototypes
    metric_results = learn_distance_metric(
        distances,
        pairs_per_prototype=params.get("pairs_per_prototype", 50),
        test_size=params.get("test_size", 0.25),
    )
    logger.info(f"Distance metric learning score: {metric_results.score}")
    return metric_results.metric_components


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


def apply_day_typing(data, parameters):
    X = data["consumption"]
    params = parameters["apply_daytyping"]

    window = int(params.get("window", 24))
    n_jobs = params.get("n_jobs", 1)

    profile = get_matrix_profile(X, window=window)
    intervals = create_time_intervals(window)

    parallel = Parallel(n_jobs=n_jobs, verbose=False)
    results = parallel(
        delayed(apply_metric_learning)(X, profile, start_time, end_time, parameters)
        for (start_time, end_time) in intervals
    )

    distance_metrics = {}
    for (start_time, end_time), components in zip(intervals, results):
        distance_metrics[start_time] = {
            "end_time": end_time,
            "metric_components": components,
        }
    return distance_metrics
