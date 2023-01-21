# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
from collections import defaultdict
from typing import Callable, Dict, Literal, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype as is_float

from .validation import validate_dataset

logger = logging.getLogger("eensight")


def get_hourly_timesteps(index):
    """Get the time steps in the provided index with duration less than one day.

    Args:
        index (pd.DatetimeIndex): The input index.

    Returns:
        pd.TimedeltaIndex: The identified time steps.
    """

    if isinstance(index, pd.DatetimeIndex):
        index = index.to_series()

    time_steps = np.array(
        [pd.Timedelta(x) for x in index.diff().unique() if pd.notnull(x)]
    )
    return time_steps[time_steps < pd.Timedelta(days=1)]


def get_time_step(index: pd.DatetimeIndex):
    """Get the most frequent time step of the provided index.

    Args:
        index (pd.DatetimeIndex): The input index.

    Returns:
        pd.Timedelta: The identified time step.
    """
    dt = index.to_series().diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].mode()[0]
    return time_step if (time_step < pd.Timedelta(days=1)) else pd.Timedelta(days=1)


def _align_by_distance(data, index, tolerance):
    data = pd.merge_asof(
        pd.DataFrame(index=index),
        data,
        left_index=True,
        right_index=True,
        tolerance=tolerance,
        direction="nearest",
    )
    return data


def _align_by_value(data, index, tolerance):
    full_index = (
        pd.concat((index.to_series(), data.index.to_series()), ignore_index=True)
        .sort_values()
        .drop_duplicates()
    )

    time_step = pd.Timedelta(
        minutes=np.gcd.reduce(
            [int(delta.seconds / 60) for delta in get_hourly_timesteps(full_index)]
        )
    )

    full_index = pd.date_range(
        start=datetime.datetime.combine(full_index.min().date(), datetime.time(0, 0)),
        end=datetime.datetime.combine(
            full_index.max().date() + datetime.timedelta(days=1), datetime.time(0, 0)
        ),
        freq=time_step,
        inclusive="left",
    )

    data = data.reindex(full_index)
    data = data.interpolate(
        method="slinear",
        limit=int(tolerance / time_step),
        limit_area="inside",
        limit_direction="both",
        axis=0,
    ).reindex(index)

    return data


def align_to_index(
    data: pd.DataFrame,
    index: pd.DatetimeIndex,
    mode: Union[
        Literal["value", "distance"], Dict[str, Literal["value", "distance"]]
    ] = None,
    tolerance: Union[pd.Timedelta, Dict[str, pd.Timedelta]] = None,
    cumulative: Union[bool, Dict[str, bool]] = False,
) -> pd.DataFrame:
    """Aling the index of the input dataframe to the provided index.

    Args:
        data (pandas.DataFrame): The dataframe to align. The dataframe's index
            must be a pandas DatetimeIndex and in increasing order.
        index (pandas.DatetimeIndex): The index with which the input dataframe will be
            aligned.
        mode ({'value', 'distance'} or dict, optional): Defines how index alignment should
            be carried out. `mode='value'` interpolates the data to match the primary index,
            while `mode='distance'` matches the primary index on the nearest key of the feature's
            index. If the input dataframe has more than one columns, `mode` can be a dictionary
            that maps feature names to `mode` values. Defaults to None.
        tolerance (pandas.Timedelta or dict, optional): If `mode='value'`, `tolerance` dictates
            how far the interpolation can reach. As an example, outdoor temperature changes slowly,
            so interpolating 2 or 3 hours ahead should not be a problem. If `mode='distance'`,
            `tolerance` is the maximum time distance to match a timestamp of the input dataframe
            with a timestamp in the primary index. If the input dataframe has more than one
            columns, `tolerance` can be a dictionary that maps feature names to tolerance
            values. Defaults to None.
        cumulative (bool or dict, optional): If (a) `cumulative=True` and (b) the index of the input
            dataframe and the `index` are misaligned, and (c) the data is numerical, the data will
            be aligned after `cumsum` has been applied on it. If the input dataframe has more than
            one columns, `cumulative` can be a dictionary that maps feature names to cumulative
            values. Defaults to False.

    Returns:
        pandas.DataFrame: The input dataframe with aligned index.

    Note:
        If a feature that is both numerical and cumulative must be aligned, it will be normalized
        by dividing its value at any given timestamp by the number of minutes between this timestamp
        and the immediately preceding one (so that to avoid misleading values when the primary index
        has time gaps).
    """
    if (not isinstance(data.index, pd.DatetimeIndex)) or (
        not isinstance(index, pd.DatetimeIndex)
    ):
        raise ValueError(
            "Both the input dataframe's index and the `index` to align to must be"
            " pandas DatetimeIndex."
        )

    if (not data.index.is_monotonic_increasing) or (not index.is_monotonic_increasing):
        raise ValueError(
            "Both the input dataframe's index and the `index` to align to must be "
            "in increasing order."
        )

    data_time_step = get_time_step(data.index)
    index_time_step = get_time_step(index)

    if not isinstance(mode, dict):
        mode = {col: mode for col in data.columns}
    if not isinstance(tolerance, dict):
        tolerance = {col: tolerance for col in data.columns}
    if not isinstance(cumulative, dict):
        cumulative = {col: cumulative for col in data.columns}

    if data_time_step == index_time_step:  # aligned
        return pd.DataFrame(index=index).join(data, how="left")

    elif (
        (data_time_step < pd.Timedelta(days=1))
        and (index_time_step < pd.Timedelta(days=1))
    ) or (
        (data_time_step >= pd.Timedelta(days=1))
        and (index_time_step >= pd.Timedelta(days=1))
    ):
        to_merge = []
        for col in data.columns:
            tolerance_col = tolerance.get(col, None)
            mode_col = mode.get(col, None)

            if tolerance_col and mode_col:
                if is_float(data[col]) and cumulative.get(col, False):
                    aligned = (
                        _align_by_value(data[col].cumsum(), index, tolerance_col).diff()
                        if mode[col] == "value"
                        else _align_by_distance(
                            data[col].cumsum(), index, tolerance_col
                        ).diff()
                    )
                    aligned = aligned.div(
                        aligned.index.to_series()
                        .diff()
                        .map(lambda x: x.total_seconds() / 60),
                        axis=0,
                    )
                    logger.info(
                        f"Cumulative feature {col} normalized by time step length."
                    )
                    to_merge.append(aligned)

                elif is_float(data[col]):
                    aligned = (
                        _align_by_value(data[col], index, tolerance_col)
                        if mode[col] == "value"
                        else _align_by_distance(data[col], index, tolerance_col)
                    )
                    to_merge.append(aligned)

                else:
                    aligned = _align_by_distance(data[col], index, tolerance_col)
                    to_merge.append(aligned)

            else:
                aligned = pd.DataFrame(index=index).join(data[[col]], how="left")
                to_merge.append(aligned)

        return pd.concat(to_merge, axis=1)

    elif index_time_step < pd.Timedelta(days=1):
        full_index = (
            pd.concat((index.to_series(), data.index.to_series()), ignore_index=True)
            .sort_values()
            .drop_duplicates()
        )
        data = data.reindex(full_index).loc[full_index.min() : full_index.max()]
        for _, grouped in data.groupby(pd.Grouper(freq=data_time_step)):
            data.loc[grouped.index] = grouped.fillna(method="ffill").fillna(
                method="bfill"
            )
        return data.reindex(index)

    else:
        raise NotImplementedError(
            "Aligning sub-daily data to a daily index is not supported."
        )


def merge_data_list(
    data_list: list,
    primary: pd.DatetimeIndex,
    mode: Dict[str, Literal["value", "distance"]] = None,
    tolerance: Dict[str, pd.Timedelta] = None,
    cumulative: Dict[str, bool] = False,
):
    """Merge together all dataframes in the provided list.

    The dataframes should not have column names in common.

    Args:
        data_list (list of pandas DataFrames): The list of dataframes to merge.
        primary (pandas.DatetimeIndex): The shared index (i.e. the index to align all
            partitioned datasests with).
        mode (dict, optional): Dictionary that maps feature names to how index alignment
            should be carried out for them. 'value' interpolates the data to match the
            primary index, while 'distance' matches the primary index on the nearest key
            of the feature's index. It is needed when the index of one or more of the
            dataframes and the `primary` index are misaligned. Defaults to None.
        tolerance (dict, optional): Dictionary that maps feature names to: (a) how far
            interpolation can reach if the corresponing mode is 'value' or (b) the maximum
            time distance to match a timestamp of the input dataframe with a timestamp in
            the primary index if the corresponing mode is 'distance'. It is needed when the
            index of one or more of the dataframes and the `primary` index are misaligned.
            Defaults to None.
        cumulative (dict, optional): Dictionary that maps feature names to boolean values
            indicating whether the data is cumulative or not. Defaults to False.

    Returns:
        pandas.DataFrame: The merged dataframe.
    """

    result = None
    for data in data_list:
        result = pd.concat(
            [
                result,
                align_to_index(
                    data, primary, mode=mode, tolerance=tolerance, cumulative=cumulative
                ),
            ],
            axis=1,
        )
    return result


def validate_partitions(
    partitioned_data: Dict[str, Callable],
    rebind_names: Dict[str, str] = None,
    date_format: str = "%Y-%m-%d %H:%M:%S",
    threshold: float = 0.25,
    primary: pd.DatetimeIndex = None,
    mode: Dict[str, Literal["value", "distance"]] = None,
    tolerance: Dict[str, pd.Timedelta] = None,
    cumulative: Dict[str, bool] = False,
) -> pd.DataFrame:
    """Validate and merge input data partitions into one pandas DataFrame.

    Args:
        partitioned_data: A dictionary with file names as keys and load functions as
            values. Each load function returns a pandas DataFrame.
        rebind_names (dict, optional): Dictionary of key/value pairs used to redefine
            column names for the input dataset. Defaults to None.
        date_format (str, optional): The strftime to parse the timestamp dates. Defaults
            to %Y-%m-%d %H:%M:%S.
        threshold (float, optional): If the range of the values that share a timestamp
            is less than `threshold` times the standard deviation of the data, they are
            replaced by their average. Otherwise, they are treated as missing values.
            Defaults to 0.25.
        primary (pandas.DatetimeIndex, optional): The shared index (i.e. the index to
            align all partitioned datasests with). It is required if the partitions
            include dataframes of different features (instead of only different time
            periods of one feature). Defaults to None.
        mode (dict, optional): Dictionary that maps feature names to how index alignment
            should be carried out for them. 'value' interpolates the data to match the
            primary index, while 'distance' matches the primary index on the nearest key
            of the feature's index. It is needed when the index of one or more of the
            dataframes and the `primary` index are misaligned. If `primary` is None, it
            will be ignored.Defaults to None.
        tolerance (dict, optional): Dictionary that maps feature names to: (a) how far
            interpolation can reach if the corresponing mode is 'value' or (b) the maximum
            time distance to match a timestamp of the input dataframe with a timestamp in
            the primary index if the corresponing mode is 'distance'. It is needed when the
            index of one or more of the dataframes and the `primary` index are misaligned.
            If `primary` is None, it will be ignored. Defaults to None.
        cumulative (dict, optional): Dictionary that maps feature names to boolean values
            indicating whether the data is cumulative or not. If `primary` is None, it will
            be ignored. Defaults to False.

    Returns:
        pandas.DataFrame: Dataframe with all loaded partitions merged.
    """
    datasets = defaultdict(list)
    for _, load_func in partitioned_data.items():
        df = load_func()
        df = validate_dataset(
            df,
            rebind_names=rebind_names,
            date_format=date_format,
            threshold=threshold,
        )
        datasets[tuple(set(df.columns))].append(df)

    if (len(datasets) > 1) and (primary is None):
        raise ValueError(
            "`primary` index is required if the partitions "
            "include dataframes of different features."
        )

    horizontal_data = []
    for items in datasets.values():
        vertical_data = pd.concat(items, axis=0)
        if vertical_data.index.has_duplicates:
            raise ValueError(
                "Partitions contain overlapping datetime indices. "
                "This leads to duplicate index entries when merging."
            )
        horizontal_data.append(vertical_data.sort_index())

    if len(horizontal_data) == 1:
        if primary is not None:
            return align_to_index(
                horizontal_data[0],
                primary,
                mode=mode,
                tolerance=tolerance,
                cumulative=cumulative,
            )
        return horizontal_data[0]

    return merge_data_list(
        horizontal_data, primary, mode=mode, tolerance=tolerance, cumulative=cumulative
    )
