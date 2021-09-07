# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import pandas as pd

from eensight.utils import as_series, get_categorical_cols

from .decompose import decompose_consumption, decompose_temperature
from .holidays import add_holidays
from .nan_imputation import linear_impute
from .outlier_detection import (
    global_filter,
    global_outlier_detect,
    local_outlier_detect,
)
from .validation import (
    apply_rebind_names,
    check_column_values_not_null,
    validate_dataset,
    validate_partitions,
)

logger = logging.getLogger("preprocess-stage")


def validate_input_data(input_data, rebind_names, location):
    if isinstance(input_data, dict):
        input_data = validate_partitions(input_data, rebind_names)
    else:
        input_data = apply_rebind_names(input_data, rebind_names)
        input_data = validate_dataset(input_data)

    if ("holiday" not in input_data) and location and any(location.values()):
        country = location.get("country")
        province = location.get("province")
        state = location.get("state")
        input_data = add_holidays(input_data, country, province=province, state=state)

    for col in get_categorical_cols(input_data, int_is_categorical=False):
        input_data[col] = input_data[col].fillna(value="_novalue_")

    return input_data


############################################################################################
# Outlier identification
############################################################################################


def find_outliers(data, parameters):
    for col in parameters["find_outliers_for"]:
        params = parameters["global_filter"][col]
        data[col] = global_filter(
            data[col],
            no_change_window=params.get("no_change_window"),
            min_value=params.get("min_value"),
            max_value=params.get("max_value"),
            allow_zero=params.get("allow_zero"),
            allow_negative=params.get("allow_negative"),
        )

        resid = None
        if col == "consumption":
            params = parameters["decompose_consumption"]
            results = decompose_consumption(
                data["consumption"].dropna().to_frame("consumption"),
                dt=params["dt"],
                add_trend=params["add_trend"],
                alpha=params["alpha"],
                return_conditions=params["return_conditions"],
                return_model=params["return_model"],
            )
            resid = results.transformed["resid"]

        if col == "temperature":
            params = parameters["decompose_temperature"]
            results = decompose_temperature(
                data["temperature"].dropna().to_frame("temperature"),
                dt=params["dt"],
                alpha=params["alpha"],
                return_model=params["return_model"],
            )
            resid = results.transformed["resid"]

        if resid is None:
            resid = data[col].dropna()

        params = parameters["global_outlier_detect"]
        outliers_global = global_outlier_detect(resid, params["c"])
        params = parameters["local_outlier_detect"]
        outliers_local = local_outlier_detect(resid, params["min_samples"], params["c"])
        outliers = np.logical_and(outliers_global, outliers_local)

        data[f"{col}_outlier"] = outliers
        data[f"{col}_outlier"] = data[f"{col}_outlier"].fillna(value=False)

    return data


############################################################################################
# Impute missing data
############################################################################################


def outlier_to_nan(data):
    columns = data.filter(like="outlier", axis=1).columns
    to_drop = []

    for col in columns:
        feature, _ = col.split("_")
        if feature != "consumption":
            data[feature] = data[feature].mask(data[col], np.nan)
            to_drop.append(col)

    data = data.drop(to_drop, axis=1)
    return data


def linear_inpute_missing(data, parameters):
    for col, params in parameters["linear_impute"].items():
        data[col] = linear_impute(data[col], window=params["window"])
    return data


############################################################################################
# Check data adequacy
############################################################################################


def check_data_adequacy(data):
    missing_condition = (
        data["consumption_outlier"]
        | data["consumption"].isna()
        | data["temperature"].isna()
    )

    missing = data[["consumption"]].mask(missing_condition, np.nan)
    avail_data = dict()

    for month_year, group in missing.groupby([lambda x: x.year, lambda x: x.month]):
        check = check_column_values_not_null(
            data=group, column="consumption", mostly=0.9
        )
        avail_data[month_year] = check.result["unexpected_percent"]

    avail_data = {f"{key[0]}M{key[1]}": val for key, val in avail_data.items()}
    avail_data = pd.DataFrame.from_dict(
        avail_data, orient="index", columns=["availability"]
    )
    insufficient = avail_data[avail_data["availability"] > 0.1]

    if len(insufficient) > 0:
        logger.warning(
            f"Months with not enough data are:\n {as_series(insufficient).to_dict()}"
        )
    return avail_data