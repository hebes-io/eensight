# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import stumpy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import Bunch


def get_matrix_profile(X: pd.Series, window=24, task_type=None) -> pd.DataFrame:
    """Calculate the matrix profile of a time series.

    Args:
        X (pandas.Series): Input time series.
        window (int, optional): Window size in hours. Defaults to 24.
        task_type ({'CPU', 'GPU'}, optional): The processing unit type to use. If
            None, the CPU will be used. If GPU is selected, a CUDA driver must have
            been installed. Defaults to None.

    Raises:
        ValueError: If input data is not a pandas Series.

    Returns:
        pd.DataFrame: The first column (nnd) consists of the matrix profile,
            the second column (nnidx) consists of the matrix profile indices,
            the third column (idx) consists of the observations indices.
    """
    if not isinstance(X, pd.Series):
        raise ValueError("This function expects pd.Series as an input")

    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta("1H") / time_step)

    m = window * steps_per_hour
    if task_type == "GPU":
        mp = stumpy.gpu_stump(X, m)
    else:
        mp = stumpy.stump(X, m)

    profile = pd.DataFrame(
        data=mp[:, :2], index=X.index[: -m + 1], columns=["nnd", "nnidx"]
    )
    profile["idx"] = profile.reset_index().index.astype(np.int32)
    profile["nnd"] = profile["nnd"].astype(float)
    profile["nnidx"] = profile["nnidx"].astype(np.int32)
    return profile  # matrix profile


def _maximum_mean_discrepancy(data):
    Kyy = euclidean_distances(data)
    ny = data.shape[0]
    data_term = (np.sum(Kyy) - np.trace(Kyy)) / (ny * (ny - 1))

    def calculate_mmd(prototypes):
        nx = prototypes.shape[0]
        Kxx = euclidean_distances(prototypes)
        Kxy = euclidean_distances(prototypes, data)

        t1 = (np.sum(Kxx) - np.trace(Kxx)) / (nx * (nx - 1))
        t2 = 2.0 * np.mean(Kxy)

        return (t1 - t2 + data_term) / data_term

    return calculate_mmd


def find_prototypes(
    X: pd.Series,
    mp: pd.DataFrame,
    start_time: time,
    end_time: time = None,
    max_iter: int = 30,
):
    """Find consumption profile prototypes.

    Prototypes are a small set of daily or (sub)daily profiles that adequately
    summarize the available data.

    Args:
        X (pd.Series): The input time series.
        mp (pd.DataFrame): The matrix profile information (output of the
            :func:`get_matrix_profile` function).
        start_time (time): The start time of the (sub)daily interval in which
            the function will search for prototypes.
        end_time (time, optional): The end time of the (sub)daily interval.
            Defaults to None.
        max_iter (int, optional): The maximum number of prototype candidates.
            Defaults to 30.

    Raises:
        ValueError: If the input data is not pandas Series.

    Returns:
        sklearn.utils.Bunch: Dict-like structure that includes the indices of
            the prototypes, the Euclidean distance of all (sub)daily profiles
            from each prototype, and the maximum mean discrepancy after including
            each of the prototype candidates.
    """
    if not isinstance(X, pd.Series):
        raise ValueError("This function expects pd.Series as an input")

    if start_time == end_time:
        end_time = None

    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta("1H") / time_step)

    if end_time is not None:
        if end_time == time(0, 0):
            date_diff = datetime.combine(
                date.today() + timedelta(days=1), end_time
            ) - datetime.combine(date.today(), start_time)
        else:
            date_diff = datetime.combine(date.today(), end_time) - datetime.combine(
                date.today(), start_time
            )
        m = int(date_diff.total_seconds() * steps_per_hour / 3600)
    else:
        m = 24 * steps_per_hour

    data = X.to_frame("values")
    data["date"] = data.index.date
    data["time"] = data.index.time

    if end_time is not None:
        data_ = data.between_time(start_time, end_time, include_end=False).pivot(
            index="date", columns="time", values="values"
        )
    else:
        data_ = data.pivot(index="date", columns="time", values="values")

    dist_mp = None
    distance_from_prototypes = []
    mp_daily = mp[mp.index.time == start_time]

    min_idx = int(mp_daily.iloc[mp_daily["nnd"].argmin()]["idx"])
    prototype = X[min_idx : min_idx + m]
    patterns = [min_idx]

    distance = pd.Series(data=stumpy.core.mass(prototype, X), index=X.index[: -m + 1])
    distance = distance[distance.index.time == start_time]
    distance_from_prototypes.append(distance.to_frame(0))
    distance = distance.to_frame("nnd")

    mmd_scores = []
    calculate_mmd = _maximum_mean_discrepancy(data_)
    prototypes = np.array(prototype).T.reshape(1, -1)

    for i in range(1, max_iter):
        if dist_mp is None:
            dist_mp = distance
        else:
            dist_mp = dist_mp.mask(distance < dist_mp, distance)

        rel_profile = (mp_daily["nnd"] / dist_mp["nnd"]).to_frame()
        rel_profile["idx"] = mp_daily["idx"]
        rel_profile["nnd"] = rel_profile["nnd"].clip(upper=1, lower=0)

        threshold = rel_profile["nnd"].quantile(0.01)
        min_idx = int(
            rel_profile[rel_profile["nnd"] <= threshold].sample()["idx"].item()
        )
        prototype = X[min_idx : min_idx + m]
        patterns.append(min_idx)

        distance = pd.Series(
            data=stumpy.core.mass(prototype, X), index=X.index[: -m + 1]
        )
        distance = distance[distance.index.time == start_time]
        distance_from_prototypes.append(distance.to_frame(i))
        distance = distance.to_frame("nnd")

        prototypes = np.concatenate((np.array(prototype).T.reshape(1, -1), prototypes))
        mmd_scores.append(calculate_mmd(prototypes))

    return Bunch(
        prototypes=patterns,
        distance_from_prototypes=pd.concat(distance_from_prototypes, axis=1),
        mmd_scores=mmd_scores,
    )
