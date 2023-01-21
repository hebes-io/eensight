# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Dict, Tuple

import pandas as pd

from eensight.methods.preprocessing.adequacy import (
    check_data_adequacy,
    expand_dates,
    filter_data,
)
from eensight.methods.preprocessing.alignment import validate_partitions
from eensight.utils import get_categorical_cols

logger = logging.getLogger("eensight")


############################################################################################
# Input data validation
############################################################################################


def validate_inputs(
    features: Dict[str, Callable],
    labels: Dict[str, Callable],
    rebind_names: Dict[str, str],
    date_format: str,
    threshold: float,
    mode: dict,
    tolerance: dict,
    cumulative: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate input data.

    Args:
        features (dictionary of `load` functions): The input feature data as a partitioned
            dataset.
        labels (dictionary of `load` functions): The input label data as a partitioned
            dataset.
        rebind_names (dict): Dictionary of key/value pairs used to redefine column names
            for the input data.
        date_format (str): The strftime to parse the timestamp dates.
        threshold (float): If the range of the values that share a timestamp is less than
            `threshold` times the standard deviation of the data, they are replaced by their
            average. Otherwise, they are treated as missing values.
        mode (dict): Dictionary that maps feature names to how index alignment should be carried
            out for them. 'value' interpolates the data to match the labels' index, while 'distance'
            matches the labels' index on the nearest key of the feature's index.
        tolerance (dict): Dictionary that maps feature names to: (a) how far interpolation can reach
            if the corresponing mode is 'value' or (b) the maximum time distance to match a timestamp
            of the features dataframes with a timestamp in the labels' index if the corresponing mode
            is 'distance'.
        cumulative (dict): Dictionary that maps feature names to boolean values indicating whether the
            corresponding data is cumulative or not.

    Returns:
        tuple containing

        - **validated_features** (*pandas.DataFrame*): The validated feature data.
        - **validated_labels** (*pandas.DataFrame*): The validated label data.
    """

    tolerance = {key: pd.Timedelta(minutes=val) for key, val in tolerance.items()}

    # start with the labels
    validated_labels = validate_partitions(
        labels,
        rebind_names=rebind_names,
        date_format=date_format,
        threshold=threshold,
        mode=mode,
        tolerance=tolerance,
        cumulative=cumulative,
    )

    if len(validated_labels.columns) > 1:
        if "consumption" in validated_labels.columns:
            validated_labels = validated_labels[["consumption"]]
        else:
            raise ValueError(
                "The `labels` dataframe must contain a column named `consumption`. "
                f"Column names found: {validated_labels.columns.to_list()}"
            )

    # proceed with the features
    validated_features = validate_partitions(
        features,
        rebind_names=rebind_names,
        date_format=date_format,
        threshold=threshold,
        primary=validated_labels.index,
        mode=mode,
        tolerance=tolerance,
        cumulative=cumulative,
    )

    # fill in missing values for categorical features
    for col in get_categorical_cols(validated_features):
        validated_features[col] = validated_features[col].fillna(value="_novalue_")

    validated_labels = validated_labels.dropna()
    validated_features = validated_features.loc[
        validated_features.index.isin(validated_labels.index)
    ]
    return validated_features, validated_labels


############################################################################################
# Data adequacy
############################################################################################


def evaluate_inputs(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    min_value: float,
    max_value: float,
    allow_zero: bool,
    allow_negative: bool,
    max_missing_pct: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate data adequacy.

    Args:
        features (pd.DataFrame): The input feature data.
        labels (pd.DataFrame): The input label data.
        min_value (float): Minumun acceptable value. Values lower than `min_value` will
            be replaced by NaN.
        max_value (float): Maximum acceptable value. Values greater than `max_value`
            will be replaced by NaN.
        allow_zero (bool): Whether zero values should be allowed.
        allow_negative (bool): Whether negative values should be allowed.
        max_missing_pct (float): The maximum acceptable percentage of missing data per month.

    Returns:
        tuple containing

        - **features** (*pandas.DataFrame*): The preprocessed feature data.
        - **labels** (*pandas.DataFrame*): The preprocessed label data.
        - **missing** (*pandas.DataFrame*): Dataframe with percentage of missing data per month.
    """

    features = expand_dates(features)
    labels = expand_dates(labels)

    labels["consumption"] = filter_data(
        labels["consumption"],
        min_value=min_value,
        max_value=max_value,
        allow_zero=allow_zero,
        allow_negative=allow_negative,
    )

    missing = check_data_adequacy(features, labels, max_missing_pct)

    labels = labels.dropna()
    features = features.loc[features.index.isin(labels.index)]
    return features, labels, missing
