# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import warnings
from functools import partial
from typing import List, Union

import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, SplineTransformer
from sklearn.utils.validation import check_is_fitted, column_or_1d
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tsmoothie.smoother import KalmanSmoother

from eensight.methods.prediction.metrics import cvrmse
from eensight.methods.preprocessing.alignment import get_time_step
from eensight.utils import as_list, get_categorical_cols, tensor_product

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)


BOOST_PARAMS = {
    "learning_rate": 0.01,
    "depth": 3,
    "bootstrap_type": "Bayesian",
    "early_stopping_rounds": 100,
    "task_type": "CPU",
    "has_time": False,
    "allow_writing_files": False,
    "verbose": False,
}


##################################################################################
# Utilities
##################################################################################


def _sigmoid(X, c1, c2):
    return 1 / (1 + np.exp(-c1 * (X - c2)))


def _rescale(X, scale_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=scale_range)

    return pd.Series(
        scaler.fit_transform(column_or_1d(X).reshape(-1, 1)).squeeze(), index=X.index
    )


def _select_c2(trial, X, y, activity):
    c1 = trial.suggest_float("c1", 5, 100, step=1)
    c2 = trial.suggest_float("c2", 0.1, 0.9)

    activity_trimmed = (
        activity.pipe(_sigmoid, c1, c2)
        .pipe(_rescale)
        .map(lambda x: 0 if x < 0.1 else 1 if x > 0.9 else x)
    )

    dmatrix = SplineTransformer(n_knots=4, degree=3, knots="quantile").fit_transform(X)

    prediction = []
    for weights in [activity_trimmed, 1 - activity_trimmed]:
        prediction.append(
            pd.Series(
                LinearRegression(fit_intercept=False)
                .fit(dmatrix, y, sample_weight=weights)
                .predict(dmatrix)
                .squeeze(),
                index=y.index,
            )
            * weights
        )

    prediction = np.sum(prediction, axis=0)
    return mean_squared_error(y["consumption"], prediction)


def _overfit_proxy(X, y, y_eval, activity, c1, c2):
    activity_trimmed = (
        activity.pipe(_sigmoid, c1, c2)
        .pipe(_rescale)
        .map(lambda x: 0 if x < 0.1 else 1 if x > 0.9 else round(x, 1))
    )

    dmatrix = tensor_product(
        SplineTransformer(n_knots=3, degree=3, knots="quantile").fit_transform(
            activity_trimmed.values.reshape(-1, 1)
        ),
        SplineTransformer(n_knots=4, degree=3, knots="quantile").fit_transform(
            X.values
        ),
    )

    prediction = pd.Series(
        LinearRegression(fit_intercept=False)
        .fit(dmatrix, y)
        .predict(dmatrix)
        .squeeze(),
        index=y.index,
    )
    return np.abs(cvrmse(y_eval, prediction) - cvrmse(y, prediction))


def _select_c1(trial, X, y, y_eval, activity, c2, init_value):
    c1 = trial.suggest_float("c1", 5, 100, step=1)
    proxy = _overfit_proxy(X, y, y_eval, activity, c1, c2)
    return ((init_value - proxy) / init_value) - 0.005 * c1


def _objective_adjust(trial, X, y, activity, predictor, upper_bound):
    true_bound = trial.suggest_float("true_bound", 0, upper_bound)
    c1 = trial.suggest_float("c1", 5, 50, step=1)
    c2 = trial.suggest_float("c2", 0.1, 0.9)

    activity_trimmed = (
        activity.pipe(_sigmoid, c1, c2)
        .pipe(_rescale, (0, true_bound))
        .map(lambda x: 0 if x < 0.1 else 1 if x > 0.9 else round(x, 1))
    )

    X_act = pd.concat([X, activity_trimmed.to_frame("activity")], axis=1)
    y_pred = predictor.predict(X_act)
    return mean_squared_error(y["consumption"], y_pred)


##################################################################################
# Activity estimation
##################################################################################


def estimate_activity_markov(
    X: pd.DataFrame,
    y: pd.DataFrame,
    exog: str = "temperature",
    n_bins: int = 5,
):
    """Estimate activity using a Markov switching regression model with 2 regimes.

    Args:
        X (pandas.DataFrame): The feature data.
        y (pandas.DataFrame): The label data.
        exog (str, optional): The name of the feature to use for the underlying
            univariate Markov switching dynamic regression. It should not contain
            categorical data. Defaults to "temperature".
        n_bins (int, optional): The number of bins to discretize the `feature` values.
            Defaults to 5.

    Returns:
        pandas.Series: The estimated activity time series.
    """

    if not pd.api.types.is_float_dtype(X[exog]):
        raise ValueError("`exog` feature cannot contain categorical data.")

    index = X.index
    X = X.dropna()
    y = y.loc[X.index]

    subsampled = False
    if get_time_step(index) < pd.Timedelta(hours=1):
        subsampled = True
        X = X.groupby([lambda x: x.date, lambda x: x.hour]).sample(n=1).sort_index()
        y = y.loc[X.index]

    enc = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="uniform")
    X = enc.fit_transform(X[[exog]])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        results = sm.tsa.MarkovRegression(
            y.values.squeeze(), k_regimes=2, exog=X, switching_variance=False
        ).fit()

    act = pd.Series(
        results.smoothed_marginal_probabilities[:, 1], index=y.index
    ).reindex(index)

    if subsampled:
        act = (
            act.groupby([lambda x: x.date, lambda x: x.hour])
            .fillna(method="ffill")
            .fillna(method="bfill")
        )

    # round up and down to remove small variations that may be picked
    # up by a predicitive model
    return act.map(lambda x: 0 if x < 0.1 else 1 if x > 0.9 else x)


def extract_activity(
    X: pd.DataFrame,
    y: pd.DataFrame,
    non_occ_features: Union[str, List[str]] = None,
    cat_features: Union[str, List[str]] = None,
    scale_range: tuple = (0, 1),
) -> pd.Series:
    """Perform the activity estimation task.

    The function uses two (2) quantile regressions of the label data as function
    of the `features` in input `X`: one at the 0.99 quantile and one at the 0.01
    one. Then, all observations are normalized to the [0, 1] interval using the
    range of the corresponding 0.99-quantile and 0.01-quantile predictions.

    Args:
        X (pandas.DataFrame): The feature data.
        y (pandas.DataFrame): The label data.
        non_occ_features (str or list of str, optional): The names of the features
            to use to fit the two quantile regression models on the label data. If
            not provided, "temperature" will be used. Defaults to None.
        cat_features (str or list of str, optional): The name(s) of the categorical
            features in the input data. Defaults to None.
        scale_range (tuple (min, max), optional): Desired range of activity levels.
            Defaults to `(0, 1)`.

    Returns:
        pandas.Series: The estimated activity time series.
    """

    a, b = scale_range
    non_occ_features = as_list(non_occ_features) or ["temperature"]
    cat_features = as_list(cat_features)

    if cat_features:
        cat_features = [name for name in cat_features if name in non_occ_features]
    else:
        cat_features = [
            name
            for name in get_categorical_cols(X, int_is_categorical=False)
            if name in non_occ_features
        ]

    model_upper = CatBoostRegressor(
        **dict(
            BOOST_PARAMS,
            loss_function="Quantile:alpha=0.99",
            iterations=600,
            cat_features=cat_features,
        )
    )
    model_lower = CatBoostRegressor(
        **dict(
            BOOST_PARAMS,
            loss_function="Quantile:alpha=0.01",
            iterations=200,
            cat_features=cat_features,
        )
    )

    model_upper = model_upper.fit(X[non_occ_features], y["consumption"])
    model_lower = model_lower.fit(X[non_occ_features], y["consumption"])

    pred_upper = pd.Series(model_upper.predict(X[non_occ_features]), index=X.index)
    pred_lower = pd.Series(model_lower.predict(X[non_occ_features]), index=X.index)

    act = (b - a) * ((y["consumption"] - pred_lower) / (pred_upper - pred_lower)) + a
    act = act.clip(lower=a, upper=b)
    return act


def estimate_activity(
    X: pd.DataFrame,
    y: pd.DataFrame,
    non_occ_features: Union[str, List[str]] = None,
    cat_features: Union[str, List[str]] = None,
    exog: str = "temperature",
    n_trials: int = 200,
    assume_hurdle: bool = False,
    verbose: bool = False,
    return_params: bool = False,
):
    """Perform the activity estimation task.

    Args:
        X (pandas.DataFrame): The feature data.
        y (pandas.DataFrame): The label data.
        non_occ_features (str or list of str, optional): The names of the features to use
            to fit the two quantile regression models on the label data. If not provided,
            "temperature" will be used. Defaults to None.
        cat_features (str or list of str, optional): The name(s) of the categorical
            features in the input data. Defaults to None.
        exog (str, optional): The name of the feature to use for determining how to best split
            the data into two regimes. It should not contain categorical data. Defaults to
            "temperature".
        n_trials (int, optional): The number of iterations for the underlying optimization.
            Defaults to 200.
        assume_hurdle (bool, optional): If True, it is assumed that the energy consumption
            data is generated from a hurdle model. A hurdle model is a two-part model that
            specifies one process for zero values and another process for the positive ones.
            Defaults to False.
        verbose (bool, optional): If True, a progress bar is visible. Defaults to False.
        return_params (bool, optional): If True, a dictionary with the optimal parameters of
            the sigmoid transformation will be returned as second argument. Defaults to False.

    Returns:
        pandas.Series: The estimated activity time series.
    """

    cat_features = as_list(cat_features)
    cat_features = [name for name in cat_features if name in X.columns]

    if assume_hurdle:
        y_on = y[y["consumption"] > 1e-05]
        X_on = X.loc[y_on.index]

        act = (
            extract_activity(
                X_on,
                y_on,
                non_occ_features=non_occ_features,
                cat_features=cat_features,
                scale_range=(0.05, 1),
            )
            .map(lambda x: round(x, 1))
            .reindex(y.index)
            .fillna(value=0)
        )
    else:
        act = extract_activity(
            X, y, non_occ_features=non_occ_features, cat_features=cat_features
        ).map(lambda x: round(x, 1))

    X = X.dropna()
    _objective = partial(
        _select_c2,
        X=X[[exog]],
        y=y.loc[X.index],
        activity=act.loc[X.index],
    )

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
        direction="minimize",
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=verbose)
    c2 = study.best_params["c2"]

    # denoise the energy consumption data
    smoother = KalmanSmoother(component="level", component_noise={"level": 0.1})
    smoother = smoother.smooth(y)
    y_sm = pd.DataFrame(smoother.smooth_data[0], index=y.index, columns=["consumption"])
    y_sm.loc[y[y["consumption"] <= 1e-05].index] = 0

    init_value = _overfit_proxy(
        X[[exog]],
        y.loc[X.index],
        y_sm.loc[X.index],
        act.loc[X.index],
        5,
        c2,
    )
    _objective = partial(
        _select_c1,
        X=X[[exog]],
        y=y.loc[X.index],
        y_eval=y_sm.loc[X.index],
        activity=act.loc[X.index],
        c2=c2,
        init_value=init_value,
    )

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        direction="maximize",
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=verbose)
    c1 = study.best_params["c1"]

    act = (
        act.pipe(_sigmoid, c1, c2)
        .pipe(_rescale)
        .map(lambda x: 0 if x < 0.1 else 1 if x > 0.9 else round(x, 1))
    )

    if return_params:
        return act, {"c1": c1, "c2": c2}
    return act


def adjust_activity(
    X: pd.DataFrame,
    y: pd.DataFrame,
    predictor: BaseEstimator,
    non_occ_features: Union[str, List[str]] = None,
    cat_features: Union[str, List[str]] = None,
    assume_hurdle: bool = False,
    n_trials: int = 500,
    upper_bound: float = 2,
    verbose: bool = False,
    return_params: bool = False,
):
    """Adjust activity levels for events that affect density of energy consumption.

    Args:
        X (pandas.DataFrame): The feature data.
        y (pandas.DataFrame): The label data.
        predictor (BaseEstimator): A trained model that predicts energy consumption given
            `non_occ_features` and activity levels.
        non_occ_features (str or list of str, optional): The names of the occupancy-
            independent features to use for estimating activity levels. If not provided,
            "temperature" will be used. Defaults to None.
        cat_features (str or list of str, optional): The name(s) of the categorical
            features in the input data. Defaults to None.
        assume_hurdle (bool, optional): If True, it is assumed that the energy consumption
            data is generated from a hurdle model. Defaults to False.
        n_trials (int, optional): The number of iterations for the underlying optimization.
            Defaults to 500.
        upper_bound (float, optional): The maximum value that the true upper bound for the
            adjusted activity levels can take. Defaults to 2.
        verbose (bool, optional): If True, a progress bar is visible. Defaults to False.
        return_params (bool, optional): If True, a dictionary with the optimal parameters of
            the sigmoid transformation will be returned as second argument. Defaults to False.

    Returns:
        pandas.Series: The adjusted activity time series.
    """

    check_is_fitted(predictor)
    non_occ_features = as_list(non_occ_features) or ["temperature"]

    if assume_hurdle:
        y_on = y[y["consumption"] > 1e-05]
        X_on = X.loc[y_on.index]

        activity = (
            extract_activity(
                X_on,
                y_on,
                non_occ_features=non_occ_features,
                cat_features=cat_features,
                scale_range=(0.05, 1),
            )
            .map(lambda x: round(x, 1))
            .reindex(y.index)
            .fillna(value=0)
        )
    else:
        activity = extract_activity(
            X, y, non_occ_features=non_occ_features, cat_features=cat_features
        ).map(lambda x: round(x, 1))

    _objective = partial(
        _objective_adjust,
        X=X,
        y=y,
        activity=activity,
        predictor=predictor,
        upper_bound=upper_bound,
    )
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=verbose)

    c1 = study.best_params["c1"]
    c2 = study.best_params["c2"]
    true_bound = study.best_params["true_bound"]

    activity = (
        activity.pipe(_sigmoid, c1, c2)
        .pipe(_rescale, (0, true_bound))
        .map(lambda x: 0 if x < 0.1 else 1 if x > 0.9 else round(x, 1))
    )

    if return_params:
        return activity, {
            "c1": c1,
            "c2": c2,
            "true_bound": true_bound,
        }
    return activity
