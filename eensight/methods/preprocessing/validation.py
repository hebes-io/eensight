# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import datetime
import decimal
import json
import logging
import sys
import traceback
from typing import Dict

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

logger = logging.getLogger("eensight")


############################################################################################
# Utility functions

# Adapted from https://github.com/great-expectations/great_expectations
############################################################################################


def convert_to_json_serializable(data):
    """Helper function to convert an object to one that is json serializable.

    Args:
        data (Any): An object to attempt to convert a corresponding json-serializable
            object

    Returns:
        dict: The converted object.
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
    """Helper function to check if an object is json serializable.

    Args:
        data (Any): An object to attempt to convert into a json-serializable
            object.
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
    def __init__(self, success=None, result=None, exception_info=None):
        self.success = success
        self.result = result or {}

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
            "exception_info": convert_to_json_serializable(self.exception_info),
        }


############################################################################################
# Validation checks
############################################################################################


def check_column_exists(
    data: pd.DataFrame,
    column: str,
    column_index: int = None,
    catch_exceptions: bool = None,
):
    """Expect the specified column to exist.

    Args:
        data (pandas DataFrame): The data to validate.
        column (str): The column name.
        column_index (int or None): If not None, checks also for the location column_index
            (zero-indexed).
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.

    Returns:
        ValidationResult
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
                success=False, result=None, exception_info=exception_info
            )
        else:
            raise ex

    else:
        return ValidationResult(
            success=bool(success), result=None, exception_info=exception_info
        )


def check_column_values_unique(
    data: pd.DataFrame,
    column: str,
    catch_exceptions: bool = None,
):
    """Expect the specified column's values to be unique.

    Args:
        data (pandas DataFrame): The data to validate.
        column (str): The column name.
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.

    Returns:
        ValidationResult
    """
    exception_info = None

    try:
        n_obs = len(data[column])
        n_missing = data[column].isna().sum()
        count = n_obs - data[column].nunique()
        success = count == 0

    except Exception as ex:
        if catch_exceptions:
            exceptiondata = traceback.format_exc().splitlines()
            exception_info = {
                "raised_exception": True,
                "where": exceptiondata[1],
                "why": exceptiondata[-1],
            }
            return ValidationResult(
                success=False, result=None, exception_info=exception_info
            )
        else:
            raise ex

    else:
        result = {
            "element_count": n_obs,
            "missing_count": n_missing,
            "unexpected_count": count,
        }
        return ValidationResult(
            success=bool(success),
            result=result,
            exception_info=exception_info,
        )


def check_column_values_increasing(
    data: pd.DataFrame,
    column: str,
    parse_as_datetimes: bool = False,
    catch_exceptions: bool = None,
):
    """Expect column values to be increasing.

    Args:
        data (pandas DataFrame): The data to validate.
        column (str): The column name.
        parse_as_datetimes (boolean or None) : If True, column values will be parsed to
            datetimes before making comparisons
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.

    Returns:
        ValidationResult
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
                success=False, result=None, exception_info=exception_info
            )
        else:
            raise ex

    else:
        return ValidationResult(
            success=bool(success), result=None, exception_info=exception_info
        )


def check_column_type_datetime(
    data: pd.DataFrame, column: str, catch_exceptions: bool = None
):
    """Expect column values to be of datetime type.

    Args:
        data (pandas DataFrame): The data to validate.
        column (str): The column name.
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.

    Returns:
        ValidationResult
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
                success=False, result=None, exception_info=exception_info
            )
        else:
            raise ex

    else:
        return ValidationResult(
            success=bool(success), result=None, exception_info=exception_info
        )


def check_column_values_not_null(
    data: pd.DataFrame,
    column: str,
    catch_exceptions: bool = None,
):
    """Expect column values to not be null.

    Args:
        data (pandas DataFrame): The data to validate.
        column (str): The column name.
        catch_exceptions (boolean or None): If True, then catch exceptions and include them
            as part of the result object.

    Returns:
        ValidationResult
    """
    exception_info = None
    expected_column = data[column]

    try:
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
                success=False, result=None, exception_info=exception_info
            )
        else:
            raise ex

    else:
        success = count == 0
        result = {
            "element_count": n_obs,
            "missing_count": count,
            "unexpected_count": count,
        }
        return ValidationResult(
            success=bool(success),
            result=result,
            exception_info=exception_info,
        )


############################################################################################
# Validation corrective actions
############################################################################################


def remove_duplicate_dates(
    data: pd.DataFrame, date_column: str, threshold: float = 0.25
):
    """Remove duplicate timestamps.

    Args:
        data (pandas.DataFrame): The data to correct.
        date_column (str): The name of the column that contains datetime information.
        threshold (float, optional): If the range of the values that share a timestamp is less
            than `threshold` times the standard deviation of the data, they are replaced by their
            average. Otherwise, they are treated as missing values. Defaults to 0.25.

    Returns:
        pandas.DataFrame: The input data without the duplicate timestamps.
    """

    results = []
    if not is_datetime(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])

    data = data.set_index(date_column)

    for col in data.columns:
        data_ = data[[col]]
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
            dupl_range["range"] < threshold * np.std(data_).item()
        ].index
        fill_values = dupl_range.loc[fill_index]["mean"].values
        data_.loc[data_.index.isin(fill_index)] = fill_values.reshape(-1, 1)
        results.append(data_)

    return pd.concat(results, axis=1).reset_index()


def apply_rebind_names(data: pd.DataFrame, rebind_names: Dict[str, str]):
    """Rename the input dataframe columns.

    Args:
        data (pandas.DataFrame): Input dataset.
        rebind_names (dict): Dictionary of key/value pairs used to redefine
            column names for the input dataset.

    Returns:
        pandas.DataFrame: The input dataframe with renamed columns.
    """
    if rebind_names and any(rebind_names.values()):
        columns = {}
        for key, val in rebind_names.items():
            if (val is not None) and (val in data.columns):
                columns[val] = key
        data = data.rename(columns=columns)

    if np.any(data.columns.duplicated()):
        raise ValueError(
            "Duplicate column names found: "
            f"{data.columns[data.columns.duplicated()].tolist()}"
        )

    return data


############################################################################################
# Composite functions
############################################################################################


def validate_dataset(
    data: pd.DataFrame,
    rebind_names: Dict[str, str] = None,
    date_format: str = "%Y-%m-%d %H:%M:%S",
    threshold: float = 0.25,
) -> pd.DataFrame:
    """Validate a dataset.

    Args:
        data (pandas.DataFrame): The data to validate.
        rebind_names (dict, optional): Dictionary of key/value pairs used to redefine column
            names for the input dataset. Defaults to None.
        date_format (str, optional): The strftime to parse the timestamp dates. Defaults to
            %Y-%m-%d %H:%M:%S.
        threshold (float, optional): If the range of the values that share a timestamp is less
            than `threshold` times the standard deviation of the data, they are replaced by their
            average. Otherwise, they are treated as missing values. Defaults to 0.25.

    Returns:
        pandas.DataFrame: The validated dataframe.
    """
    if rebind_names:
        data = apply_rebind_names(data, rebind_names)

    if not check_column_exists(data, "timestamp").success:
        raise ValueError(f"Column `timestamp` is missing from input dataset")

    if not check_column_type_datetime(data, "timestamp").success:
        try:
            data["timestamp"] = pd.to_datetime(data["timestamp"], format=date_format)
            data = data.dropna(subset="timestamp")
        except ValueError:
            raise ValueError(f"Column `timestamp` must be in datetime format")

    if not check_column_values_increasing(data, "timestamp").success:
        data = data.sort_values(by=["timestamp"])

    if not check_column_values_unique(data, "timestamp").success:
        data = remove_duplicate_dates(data, "timestamp", threshold=threshold)

    data = data.set_index("timestamp")
    to_drop = data.filter(like="Unnamed", axis=1).columns
    if len(to_drop) > 0:
        data = data.drop(to_drop, axis=1)

    return data
