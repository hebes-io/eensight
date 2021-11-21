# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
from feature_encoders.utils import as_series, get_categorical_cols

from eensight.utils import recover_missing_dates

from .decomposion import decompose_consumption, decompose_temperature
from .drift_detection import detect_drift
from .nan_imputation import linear_impute
from .outlier_detection import (
    global_filter,
    global_outlier_detect,
    local_outlier_detect,
)
from .validation import apply_rebind_names, validate_dataset, validate_partitions

logger = logging.getLogger("preprocess")


############################################################################################
# Input data validation
############################################################################################


def validate_input_data(
    input_data: Union[pd.DataFrame, Dict[str, Callable[[], Any]]],
    rebind_names: Dict[str, str],
):
    """Validate input data.

    Args:
        input_data (Union[pd.DataFrame, Dict[str, Callable[[], Any]]]): The
            input data as a pandas DataFrame or a partitioned dataset.
        rebind_names (dict): Dictionary of key/value pairs used to redefine
            column names for the input dataset.

    Returns:
        pandas.DataFrame: The validated data.
    """
    if isinstance(input_data, dict):
        validated_data = validate_partitions(input_data, rebind_names)
    else:
        input_data = apply_rebind_names(input_data, rebind_names)
        validated_data = validate_dataset(input_data)

    validated_data = validated_data.sort_index()

    # fill in missing values for categorical features
    for col in get_categorical_cols(validated_data, int_is_categorical=False):
        validated_data[col] = validated_data[col].fillna(value="_novalue_")

    return validated_data


############################################################################################
# Drift detection
############################################################################################


def apply_drift_detection(data: pd.DataFrame, of_drift_detection: dict):
    """Identify if and when data drift exists in the input data.

    Args:
        data (pandas.DataFrame): The input data.
        of_drift_detection (dict): The parameters of the drift detection step (typically
            read from `conf/base/parameters/preprocess.yaml`).

    Returns:
        tuple containing

        - **data_with_drift** (*pandas.DataFrame*): The data subset where drift has been detected.
        - **data** (*pandas.DataFrame*): The input data without the data with drift.
    """
    data = detect_drift(
        data,
        stat_size=of_drift_detection.get("stat_size", 150),
        stat_distance=of_drift_detection.get("stat_distance", 0.1),
        window=of_drift_detection.get("window", 300),
        alpha=of_drift_detection.get("alpha", 0.005),
        logger=logger,
    )
    data_with_drift = data[data["drift"]].drop("drift", axis=1)
    data = data[~data["drift"]].drop("drift", axis=1)
    data = recover_missing_dates(data)
    return data_with_drift, data


############################################################################################
# Outlier identification
############################################################################################


def find_outliers(
    data: pd.DataFrame,
    find_outliers_for: list,
    of_global_filter: dict,
    of_seasonal_decompose: dict,
    of_global_outlier: dict,
    of_local_outlier: dict,
    max_outlier_pct: float,
):
    """Find potential outliers in the input data.

    Args:
        data (pandas.DataFrame): The input data.
        find_outliers_for (list): List with the names of the features for which
            the outlier step should run.
        of_global_filter (dict): The parameters of the global filter step (typically
            read from `conf/base/parameters/preprocess.yaml`).
        of_seasonal_decompose (dict): The parameters of the seasonal decomposition step
            (typically read from `conf/base/parameters/preprocess.yaml`).
        of_global_outlier (dict): The parameters of the global outlier detection step
            (typically read from `conf/base/parameters/preprocess.yaml`).
        of_local_outlier (dict): The parameters of the local outlier detection step
            (typically read from `conf/base/parameters/preprocess.yaml`).
        max_outlier_pct (float): The maximum percentage of observations that can be
            marked as outliers.

    Returns:
        pandas.DataFrame: The input data with an additional feature indicating whether is marked as outlier.
    """
    for col in find_outliers_for:
        params = of_global_filter[col]
        data[col] = global_filter(
            data[col],
            no_change_window=params.get("no_change_window", 3),
            max_pct_of_dummy=params.get("max_pct_of_dummy", 0.1),
            min_value=params.get("min_value"),
            max_value=params.get("max_value"),
            allow_zero=params.get("allow_zero"),
            allow_negative=params.get("allow_negative"),
        )

        nc_score = None  # non-conformity score
        if col == "consumption":
            params = of_seasonal_decompose[col]
            try:
                results = decompose_consumption(
                    data[[col]].dropna(),
                    ds=params.get("ds"),
                    add_trend=params.get("add_trend", False),
                    min_samples=params.get("min_samples", 0.8),
                    alpha=params.get("alpha", 0.01),
                    return_model=params.get("return_model", False),
                )
            except ValueError as exc:
                message = str(exc) + " Seasonal decomposition skipped."
                logger.exception(message, exc_info=False)
            else:
                nc_score = data[col] - results.prediction[col]
                nc_score = nc_score.dropna()
                logger.info(
                    "CV(RMSE) of seasonal predictor: "
                    f"{np.sqrt(np.mean(nc_score**2)) / np.mean(data[col])}"
                )

        if col == "temperature":
            params = of_seasonal_decompose[col]
            try:
                results = decompose_temperature(
                    data[[col]].dropna(),
                    ds=params.get("ds"),
                    min_samples=params.get("min_samples", 0.8),
                    alpha=params.get("alpha", 0.01),
                    return_model=params.get("return_model", False),
                )
            except ValueError as exc:
                message = str(exc) + " Seasonal decomposition is skipped."
                logger.exception(message, exc_info=False)
            else:
                nc_score = data[col] - results.prediction[col]
                nc_score = nc_score.dropna()
                logger.info(
                    "CV(RMSE) of seasonal predictor: "
                    f"{np.sqrt(np.mean(nc_score**2)) / np.mean(data[col])}"
                )

        if nc_score is None:
            nc_score = data[col]

        outliers_global = global_outlier_detect(nc_score, of_global_outlier.get("c", 7))
        outliers_local = local_outlier_detect(
            nc_score,
            of_local_outlier.get("min_samples", 0.66),
            of_local_outlier.get("c", 8),
        )

        no_outliers = np.logical_or(outliers_global == 0, outliers_local == 0)
        outlier_score = outliers_global + outliers_local
        outlier_score = outlier_score.mask(no_outliers, 0)

        n_outliers = int(max_outlier_pct * data[col].notna().sum())
        outliers = outlier_score[outlier_score > 0].nlargest(n_outliers)

        data[f"{col}_outlier"] = False
        data.loc[outliers.index, f"{col}_outlier"] = True

    return data


############################################################################################
# Impute missing data
############################################################################################


def outlier_to_nan(data: pd.DataFrame):
    """Replace all outliers - except for consumption - by `NaN`.

    Args:
        data (pandas.DataFrame): [description]

    Returns:
        pandas.DataFrame: The input data after replacing outliers.
    """
    columns = data.filter(like="_outlier", axis=1).columns
    to_drop = []

    for col in columns:
        feature, _ = col.split("_")
        if feature != "consumption":
            data[feature] = data[feature].mask(data[col], np.nan)
            to_drop.append(col)

    data = data.drop(to_drop, axis=1)
    return data


def linear_inpute_missing(data: pd.DataFrame, of_linear_impute: dict):
    """Impute missing values using linear interpolation.

    Args:
        data (pandas.DataFrame): The input data.
        of_linear_impute (dict): The parameters of the preprocessing step (typically
            read from `conf/base/parameters/preprocess.yaml`).

    Returns:
        pandas.DataFrame: The input data after with (some or all) missing values filled.
    """
    for col, params in of_linear_impute.items():
        data[col] = linear_impute(data[col], window=params["window"])
    return data


def drop_missing_data(data: pd.DataFrame):
    """Drop timestamps where any of the features' value is missing.

    Args:
        data (pandas.DataFrame): The input data.

    Returns:
        pandas.DataFrame: The input data without missing values.
    """
    return data.dropna()


############################################################################################
# Check data adequacy
############################################################################################


def check_data_adequacy(data: pd.DataFrame, max_missing_pct: float = 0.1):
    """Check whether enough data is available for each month of the year.

    Args:
        data (pandas.DataFrame): The input data.
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

    missing_condition = data["consumption_outlier"] | data.isna().any(axis=1)
    missing = data[["consumption"]].mask(missing_condition, np.nan)
    missing_per_month = dict()

    for month, group in missing.groupby([lambda x: x.month]):
        missing_per_month[month] = (
            np.sum(
                group.groupby([lambda x: x.day, lambda x: x.hour]).count() == 0
            ).item()
            / 720
        )  # hours per month

    missing_per_month = {f"M{key}": val for key, val in missing_per_month.items()}
    missing_per_month = pd.DataFrame.from_dict(
        missing_per_month, orient="index", columns=["missing_pct"]
    )
    insufficient = missing_per_month[missing_per_month["missing_pct"] > max_missing_pct]

    if len(insufficient) > 0:
        logger.warning(
            f"Months with not enough data are:\n {as_series(insufficient).to_dict()}"
        )
    return missing_per_month
