# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Union

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from eensight.utils import get_categorical_cols

BOOST_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 10000,
    "depth": 3,
    "bootstrap_type": "Bayesian",
    "early_stopping_rounds": 100,
    "task_type": "CPU",
    "has_time": True,
    "allow_writing_files": False,
    "verbose": False,
}


def train_boost(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cat_features: Union[str, List[str]] = None,
    validation_size: float = 0.2,
    return_scores: bool = False,
):
    """Train a CatBoost regressor model.

    Args:
        X (pandas.DataFrame): Feature data.
        y (pandas.DataFrame): Target data.
        cat_features (str or list of str, optional): The name(s) of the categorical features
            in the input data. Defaults to None.
        validation_size (float, optional): The size of the validation dataset (as
            percentage of all training data) to use for identifying the optimal number
            of iterations for the Gradient Boosting model. Defaults to 0.2.
        return_scores (bool, optional): If `True`, the function will also return (as second
            result) a dictionary with the regression metrics for data used for training (`learn` key)
            and for validation (`validation` key). Defaults to False.

     Returns:
        catboost.CatBoostRegressor: The trained regressor model.

    """

    if cat_features is None:
        cat_features = get_categorical_cols(X, int_is_categorical=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        shuffle=False,
    )

    model = CatBoostRegressor(**BOOST_PARAMS).fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True,
        verbose=False,
    )

    scores = model.get_best_score()
    scores["validation"]["CVRMSE"] = (
        scores["validation"]["RMSE"] / np.array(y_val).mean()
    )

    iterations = model.get_best_iteration() if model.get_best_iteration() > 100 else 100
    model = CatBoostRegressor(**dict(BOOST_PARAMS, iterations=iterations)).fit(
        X,
        y,
        cat_features=cat_features,
        verbose=False,
    )

    scores["learn"] = model.get_best_score()["learn"]
    scores["learn"]["CVRMSE"] = scores["learn"]["RMSE"] / np.array(y).mean()

    if return_scores:
        return model, scores

    return model
