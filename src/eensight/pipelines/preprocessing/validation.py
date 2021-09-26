# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# Includes code from https://github.com/great-expectations/great_expectations

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import decimal
import json
import logging
import sys
import traceback
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from dateutil.parser import parse
from pandas.api.types import is_datetime64_any_dtype as is_datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


############################################################################################
# Utility functions
############################################################################################


def convert_to_json_serializable(data):
    """
    Helper function to convert an object to one that is json serializable
    From great_expectations.core.util
    """
    try:
        if not isinstance(data, list) and pd.isna(data):
            return None
    except TypeError:
        pass
    except ValueError:
        pass

    if isinstance(data, (str, int, float, bool)):
        # No problem to encode json
        return data

    elif isinstance(data, dict):
        new_dict = {}
        for key in data:
            new_dict[str(key)] = convert_to_json_serializable(data[key])

        return new_dict

    elif isinstance(data, (list, tuple, set)):
        new_list = []
        for val in data:
            new_list.append(convert_to_json_serializable(val))

        return new_list

    elif isinstance(data, (np.ndarray, pd.Index)):
        return [convert_to_json_serializable(x) for x in data.tolist()]

    elif data is None:
        # No problem to encode json
        return data

    elif isinstance(data, (datetime.datetime, datetime.date)):
        return data.isoformat()

    elif np.issubdtype(type(data), np.bool_):
        return bool(data)

    elif np.issubdtype(type(data), np.integer) or np.issubdtype(type(data), np.uint):
        return int(data)

    elif np.issubdtype(type(data), np.floating):
        return float(round(data, sys.float_info.dig))

    elif isinstance(data, pd.Series):
        index_name = data.index.name or "index"
        value_name = data.name or "value"
        return [
            {
                index_name: convert_to_json_serializable(idx),
                value_name: convert_to_json_serializable(val),
            }
            for idx, val in data.iteritems()
        ]

    elif isinstance(data, pd.DataFrame):
        return convert_to_json_serializable(data.to_dict(orient="records"))

    elif isinstance(data, decimal.Decimal):
        if not (-1e-55 < decimal.Decimal.from_float(float(data)) - data < 1e-55):
            logger.warning(
                f"Using lossy conversion for decimal {data} "
                "to float object to support serialization."
            )
        return float(data)

    else:
        raise TypeError(
            f"{data} is of type {type(data).__name__} which cannot be serialized."
        )


def ensure_json_serializable(data):
    """
    Helper function to convert an object to one that is json serializable
    From great_expectations.core.util
    """
    try:
        if not isinstance(data, list) and pd.isna(data):
            return
    except TypeError:
        pass
    except ValueError:
        pass

    if isinstance(data, ((str,), (int,), float, bool)):
        # No problem to encode json
        return

    elif isinstance(data, dict):
        for key in data:
            str(key)  # key must be cast-able to string
            ensure_json_serializable(data[key])
        return

    elif isinstance(data, (list, tuple, set)):
        for val in data:
            ensure_json_serializable(val)
        return

    elif isinstance(data, (np.ndarray, pd.Index)):
        _ = [ensure_json_serializable(x) for x in data.tolist()]
        return

    elif data is None:
        # No problem to encode json
        return

    elif isinstance(data, (datetime.datetime, datetime.date)):
        return

    elif np.issubdtype(type(data), np.bool_):
        return

    elif np.issubdtype(type(data), np.integer) or np.issubdtype(type(data), np.uint):
        return

    elif np.issubdtype(type(data), np.floating):
        return

    elif isinstance(data, pd.Series):
        index_name = data.index.name or "index"
        value_name = data.name or "value"
        _ = [
            {
                index_name: ensure_json_serializable(idx),
                value_name: ensure_json_serializable(val),
            }
            for idx, val in data.iteritems()
        ]
        return
    elif isinstance(data, pd.DataFrame):
        return ensure_json_serializable(data.to_dict(orient="records"))

    elif isinstance(data, decimal.Decimal):
        return

    else:
        raise Exception(
            f"{data} is of type {type(data).__name__} which cannot be serialized to json"
        )


############################################################################################
# ValidationResult
############################################################################################


class ValidationResult:
    def __init__(self, success=None, result=None, meta=None, exception_info=None):
        self.success = success

        if result is None:
            result = {}
        self.result = result

        if meta is None:
            meta = {}
        ensure_json_serializable(meta)
        self.meta = meta

        self.exception_info = exception_info or {
            "raised_exception": False,
            "where": None,
            "why": None,
        }

    def __repr__(self):
        return json.dumps(self.to_json_dict(), indent=2)

    def __str__(self):
        return json.dumps(self.to_json_dict(), indent=2)

    def to_json_dict(self):
        return {
            "success": self.success,
            "result": convert_to_json_serializable(self.result),
            "meta": convert_to_json_serializable(self.meta),
            "exception_info": convert_to_json_serializable(self.exception_info),
        }


############################################################################################
# Validation checks
############################################################################################


def check_column_exists(
    data, column, column_index=None, catch_exceptions=None, meta=None
):
    """Expect the specified column to exist.

    Args:
        data (pandas DataFrame): The data to validate
        column (str): The column name.
        column_index (int or None): If not None, checks also for the location column_index
            (zero-indexed).
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.
        meta (dict or None): A JSON-serializable dictionary that will be included in the output
            without modification.

    Returns:
        A ValidationResult.
    """
    exception_info = None

    try:
        if column_index is not None:
            success = data.columns[column_index] == column
        else:
            success = column in data.columns

    except Exception as ex:
        if catch_exceptions:
            exceptiondata = traceback.format_exc().splitlines()
            exception_info = {
                "raised_exception": True,
                "where": exceptiondata[1],
                "why": exceptiondata[-1],
            }
            return ValidationResult(
                success=False, result=None, meta=meta, exception_info=exception_info
            )
        else:
            raise ex

    else:
        return ValidationResult(
            success=bool(success), result=None, meta=meta, exception_info=exception_info
        )


def check_column_values_unique(
    data, column, mostly=None, catch_exceptions=None, meta=None
):
    """Expect the specified column's values to be unique.

    Args:
        data (pandas DataFrame): The data to validate
        column (str): The column name.
        mostly (None or a float between 0 and 1): Return `"success": True` if at least mostly
            fraction of values match the expectation.
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.
        meta (dict or None): A JSON-serializable dictionary that will be included in the output
            without modification.

    Returns:
        A ValidationResult.
    """
    exception_info = None

    try:
        n_obs = len(data[column])
        n_missing = data[column].isna().sum()

        if mostly is not None:
            if not isinstance(mostly, (int, float)):
                raise TypeError("'mostly' parameter must be an integer or float")
            if not (0 <= mostly <= 1):
                raise ValueError("'mostly' parameter must be between 0 and 1")
        else:
            mostly = 1

        if is_datetime(data[column]):
            try:
                count = (
                    data[column].diff().value_counts()[pd.Timedelta("0 days 00:00:00")]
                )
            except KeyError:
                count = 0
                success = True
            else:
                success = (count / n_obs) <= (1 - mostly)
        else:
            count = n_obs - data[column].nunique()
            success = (count / n_obs) <= (1 - mostly)

    except Exception as ex:
        if catch_exceptions:
            exceptiondata = traceback.format_exc().splitlines()
            exception_info = {
                "raised_exception": True,
                "where": exceptiondata[1],
                "why": exceptiondata[-1],
            }
            return ValidationResult(
                success=False, result=None, meta=meta, exception_info=exception_info
            )
        else:
            raise ex

    else:
        result = {
            "element_count": n_obs,
            "missing_count": n_missing,
            "unexpected_count": count,
            "unexpected_percent": count / n_obs,
            "unexpected_percent_nonmissing": count / (n_obs - n_missing),
        }
        return ValidationResult(
            success=bool(success),
            result=result,
            meta=meta,
            exception_info=exception_info,
        )


def check_column_values_increasing(
    data, column, parse_as_datetimes=False, catch_exceptions=None, meta=None
):
    """Expect column values to be increasing.

    Args:
        data (pandas DataFrame): The data to validate
        column (str): The column name.
        parse_as_datetimes (boolean or None) : If True, all non-null column values
            will be parsed to datetimes before making comparisons
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.
        meta (dict or None): A JSON-serializable dictionary that will be included in the output
            without modification.

    Returns:
        A ValidationResult.
    """
    exception_info = None

    try:
        expected_column = data[column]

        if parse_as_datetimes:
            if data[column].dtype == np.int64:
                expected_column = pd.to_datetime(data[column].astype(str))
            else:
                expected_column = pd.to_datetime(data[column])
        success = expected_column.is_monotonic_increasing

    except Exception as ex:
        if catch_exceptions:
            exceptiondata = traceback.format_exc().splitlines()
            exception_info = {
                "raised_exception": True,
                "where": exceptiondata[1],
                "why": exceptiondata[-1],
            }
            return ValidationResult(
                success=False, result=None, meta=meta, exception_info=exception_info
            )
        else:
            raise ex

    else:
        return ValidationResult(
            success=bool(success), result=None, meta=meta, exception_info=exception_info
        )


def check_column_type_datetime(data, column, catch_exceptions=None, meta=None):
    """Expect column values to be of datetime type.

    Args:
        data (pandas DataFrame): The data to validate
        column (str): The column name.
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.
        meta (dict or None): A JSON-serializable dictionary that will be included in the output
            without modification.

    Returns:
        A ValidationResult.
    """
    exception_info = None

    try:
        success = is_datetime(data[column])

    except Exception as ex:
        if catch_exceptions:
            exceptiondata = traceback.format_exc().splitlines()
            exception_info = {
                "raised_exception": True,
                "where": exceptiondata[1],
                "why": exceptiondata[-1],
            }
            return ValidationResult(
                success=False, result=None, meta=meta, exception_info=exception_info
            )
        else:
            raise ex

    else:
        return ValidationResult(
            success=bool(success), result=None, meta=meta, exception_info=exception_info
        )


def check_column_values_dateutil_parseable(
    data, column, catch_exceptions=None, meta=None
):
    exception_info = None

    try:
        try:
            data[column].map(lambda x: parse(x, fuzzy=False))
            success = True
        except ValueError:
            success = False

    except Exception as ex:
        if catch_exceptions:
            exceptiondata = traceback.format_exc().splitlines()
            exception_info = {
                "raised_exception": True,
                "where": exceptiondata[1],
                "why": exceptiondata[-1],
            }
            return ValidationResult(
                success=False, result=None, meta=meta, exception_info=exception_info
            )
        else:
            raise ex

    else:
        return ValidationResult(
            success=bool(success), result=None, meta=meta, exception_info=exception_info
        )


def check_column_values_not_null(
    data, column, mostly=None, catch_exceptions=None, meta=None
):
    """Expect column values to not be null.

    Args:
        data (pandas DataFrame): The data to validate
        column (str): The column name.
        mostly (None or a float between 0 and 1): Return `"success": True` if at least mostly
            fraction of values match the expectation.
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.
        meta (dict or None): A JSON-serializable dictionary that will be included in the output
            without modification.

    Returns:
        A ValidationResult.
    """
    exception_info = None
    expected_column = data[column]

    try:
        if mostly is not None:
            if not isinstance(mostly, (int, float)):
                raise TypeError("'mostly' parameter must be an integer or float")
            if not (0 <= mostly <= 1):
                raise ValueError("'mostly' parameter must be between 0 and 1")
        else:
            mostly = 1

        n_obs = len(expected_column)
        count = expected_column.isna().sum()

    except Exception as ex:
        if catch_exceptions:
            exceptiondata = traceback.format_exc().splitlines()
            exception_info = {
                "raised_exception": True,
                "where": exceptiondata[1],
                "why": exceptiondata[-1],
            }
            return ValidationResult(
                success=False, result=None, meta=meta, exception_info=exception_info
            )
        else:
            raise ex

    else:
        success = (count / n_obs) <= (1 - mostly)
        result = {
            "element_count": n_obs,
            "missing_count": count,
            "unexpected_count": count,
            "unexpected_percent": count / n_obs,
            "unexpected_percent_nonmissing": count / n_obs,
        }
        return ValidationResult(
            success=bool(success),
            result=result,
            meta=meta,
            exception_info=exception_info,
        )


############################################################################################
# Validation corrective actions
############################################################################################


def remove_dublicate_dates(data, date_column, threshold=None):
    if threshold is None:
        threshold = 0.2

    results = []
    data = data.set_index(date_column)

    for col in data.columns:
        data_ = data[col].to_frame(col)
        dupl_range = (
            data_[data_.index.duplicated(keep=False)]
            .groupby([lambda x: x])
            .agg([lambda x: x.max() - x.min(), lambda x: x.mean()])
        )
        dupl_range.columns = dupl_range.columns.droplevel(1)
        dupl_range.columns = ["range", "mean"]

        data_ = data_[~data_.index.duplicated(keep="first")]
        data_.loc[data_.index.isin(dupl_range.index)] = np.nan

        fill_index = dupl_range[
            dupl_range["range"] < threshold * data_.mean().item()
        ].index
        fill_values = dupl_range.loc[fill_index]["mean"].values
        data_.loc[data_.index.isin(fill_index)] = fill_values.reshape(-1, 1)
        results.append(data_)

    return pd.concat(results, axis=1).reset_index()


def expand_to_all_dates(data, date_column):
    time_step = data[date_column].diff().min()

    if time_step == pd.Timedelta("0 days 00:00:00"):
        raise ValueError("Input data contains dublicate dates")

    data = data.set_index(date_column)
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
    return data.reset_index()


def apply_rebind_names(data, rebind_names):
    if rebind_names and any(rebind_names.values()):
        columns = {}
        for key, val in rebind_names.items():
            if (val is not None) and (val in data.columns):
                columns[val] = key
        data = data.rename(columns=columns)
    return data


############################################################################################
# Composite functions
############################################################################################


def validate_dataset(data, date_col="timestamp"):
    if not check_column_exists(data, date_col).success:
        raise ValueError(f"Column {date_col} is missing from input dataset")

    if (
        not check_column_type_datetime(data, date_col).success
        and check_column_values_dateutil_parseable(data, date_col).success
    ):
        data[date_col] = data[date_col].map(pd.to_datetime)
    elif not check_column_type_datetime(data, date_col).success:
        raise ValueError(f"Column `{date_col}` must be in datetime format")

    if not check_column_values_increasing(data, date_col).success:
        data = data.sort_values(by=[date_col])

    if not check_column_values_unique(data, date_col).success:
        data = remove_dublicate_dates(data, date_col)

    data = expand_to_all_dates(data, date_col)
    data = data.set_index(date_col)
    if "Unnamed: 0" in data:
        data = data.drop("Unnamed: 0", axis=1)
    return data


def merge_on_hours(left, right):
    merged = pd.merge_asof(
        left,
        right,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("1H"),
    )
    return merged


def merge_on_days(left, right):
    if isinstance(right.index[0], datetime.datetime):
        right.index = right.index.map(lambda x: x.date)

    columns = right.columns
    for col in columns:
        right = (
            left.index.to_series()
            .map(lambda x: x.date())
            .map(lambda x: right.loc[:, col].get(x, default=np.nan))
            .to_frame(col)
        )
        left = left.join(right)
    return left


def merge_data_list(data_list, primary=None):
    if len(data_list) == 1:
        return data_list[0]

    time_steps = []
    for data in data_list:
        dt = data.index.to_series().diff()
        time_steps.append(dt.iloc[dt.values.nonzero()[0]].min())

    merged = None
    if primary is not None:
        for i, data in enumerate(data_list):
            if primary in data.columns:
                merged = data_list.pop(i)
                _ = time_steps.pop(i)
                break

    if (primary is None) or (merged is None):
        i = time_steps.index(min(time_steps))
        merged = data_list.pop(i)
        _ = time_steps.pop(i)

    for i, data in enumerate(data_list):
        if time_steps[i] < pd.Timedelta(days=1):
            merged = merge_on_hours(merged, data)
        else:
            merged = merge_on_days(merged, data)
    return merged


def validate_partitions(
    partitioned_data: Dict[str, Callable[[], Any]], rebind_names: Dict[str, str]
) -> pd.DataFrame:
    """Validate and merge and input partitions into one pandas DataFrame.

    Args:
        partitioned_data: A dictionary with partition ids as keys and load functions
            as values.
        rebind_names: Dictionary of key/value pairs used to redefine column
            names for the datasets that are inputs to this pipeline.

    Returns:
        Pandas DataFrame with all loaded partitions merged.
    """
    datasets = []
    for _, load_func in partitioned_data.items():
        df = load_func()
        df = apply_rebind_names(df, rebind_names)
        df = validate_dataset(df)
        datasets.append(df)

    return merge_data_list(datasets, primary="consumption")
