# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import namedtuple
from typing import List, Union

import emcee
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from eensight.methods.features import SafeOrdinalEncoder
from eensight.utils import as_list, get_categorical_cols

from .common import train_boost

###############################################################################################
# Utilities
###############################################################################################


def _create_lagged(X, lags):
    added_features = []
    for fea, lag_values in lags.items():
        if fea not in X.columns:
            continue
        for lag in as_list(lag_values):
            added_features.append(X[fea].shift(lag).to_frame(f"L{lag}.{fea}"))
    return pd.concat([X, *added_features], axis=1)


def _create_calendar(X):
    result = pd.DataFrame(0, columns=["hour", "day", "week"], index=X.index)
    result["hour"] = X.index.hour  # hour of day
    result["day"] = X.index.dayofweek  # day of week
    result["week"] = X.index.isocalendar().week  # week of year
    return result


def _log_prior(theta, y, priors):
    u_value, T_sp_coo, T_sp_hea, T_sb_coo, T_sb_hea, eta, ach, alpha, beta = theta

    u_value_min, u_value_max = priors.get("u_value", (0.5, 5))
    T_sp_coo_min, T_sp_coo_max = priors.get("T_sp_coo", (24, 27))
    T_sp_hea_min, T_sp_hea_max = priors.get("T_sp_hea", (20, 24))
    T_sb_coo_min, T_sb_coo_max = priors.get("T_sb_coo", (24, 35))
    T_sb_hea_min, T_sb_hea_max = priors.get("T_sb_hea", (5, 20))
    eta_min, eta_max = priors.get("eta", (0, 1))
    ach_min, ach_max = priors.get("ach", (0.3, 5))
    alpha_min, alpha_max = priors.get("alpha", (0, y.max() - y.min()))
    beta_min, beta_max = priors.get("beta", (0, y.min()))

    if (
        (u_value_min <= u_value <= u_value_max)
        and (T_sp_coo_min <= T_sp_coo <= T_sp_coo_max)
        and (T_sp_hea_min <= T_sp_hea <= T_sp_hea_max)
        and (T_sb_coo_min <= T_sb_coo <= T_sb_coo_max)
        and (T_sb_hea_min <= T_sb_hea <= T_sb_hea_max)
        and (eta_min <= eta <= eta_max)
        and (ach_min <= ach <= ach_max)
        and (alpha_min <= alpha <= alpha_max)
        and (beta_min <= beta <= beta_max)
    ):
        return 0.0

    return -np.inf


def _forward_model(theta, temperature, activity, A, V, cp, rho):
    u_value, T_sp_coo, T_sp_hea, T_sb_coo, T_sb_hea, eta, ach, alpha, beta = theta

    # air conditioning load when activity is positive
    cond_load_on = (
        A
        * u_value
        * np.maximum(temperature - T_sp_coo, T_sp_hea - temperature)
        .clip(lower=0)
        .pipe(lambda x: x.mask(activity <= 1e-05, 0))
    )

    # ventilation load
    ven_load = (
        (1 / 3600)
        * (1 - eta)
        * cp
        * rho
        * ach
        * V
        * np.maximum(temperature - T_sp_coo, T_sp_hea - temperature)
        .clip(lower=0)
        .pipe(lambda x: x.mask(activity <= 1e-05, 0))
    )

    # air conditioning load when activity is zero
    cond_load_off = (
        A
        * u_value
        * np.maximum(temperature - T_sb_coo, T_sb_hea - temperature)
        .clip(lower=0)
        .pipe(lambda x: x.mask(activity > 1e-05, 0))
    )

    # plug load
    plug_load = alpha * activity + beta

    return cond_load_on, ven_load, cond_load_off, plug_load


def _log_likelihood(theta, temperature, activity, y, A, V, cp, rho):
    _, T_sp_coo, T_sp_hea, T_sb_coo, T_sb_hea, eta, _, _, _ = theta

    if (
        (T_sp_coo < T_sp_hea)
        or (T_sb_coo < T_sb_hea)
        or (T_sp_coo > T_sb_coo)
        or (T_sp_hea < T_sb_hea)
        or (eta > 1)
        or (eta < 0)
    ):
        return -np.inf

    cond_load_on, ven_load, cond_load_off, plug_load = _forward_model(
        theta, temperature, activity, A, V, cp, rho
    )
    total_load = cond_load_on + ven_load + cond_load_off + plug_load

    return -0.5 * np.sum((y - total_load) ** 2)


def _log_probability(theta, temperature, activity, y, A, V, cp, rho, priors):
    lp = _log_prior(theta, y, priors)

    if not np.isfinite(lp):
        return -np.inf

    return lp + _log_likelihood(theta, temperature, activity, y, A, V, cp, rho)


def _run_mcmc(temperature, activity, y, *, A, V, cp, rho, run_params):
    ndim = 9  # number of parameters to estimate
    nwalkers = run_params["nwalkers"]
    priors = run_params["priors"]
    n_samples = run_params["n_samples"]
    verbose = run_params["verbose"]

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        _log_probability,
        args=(temperature, activity, y, A, V, cp, rho, priors),
    )

    u_value_min, u_value_max = priors.get("u_value", (0.5, 5))
    T_sp_coo_min, T_sp_coo_max = priors.get("T_sp_coo", (24, 27))
    T_sp_hea_min, T_sp_hea_max = priors.get("T_sp_hea", (20, 24))
    T_sb_coo_min, T_sb_coo_max = priors.get("T_sb_coo", (30, 35))
    T_sb_hea_min, T_sb_hea_max = priors.get("T_sb_hea", (5, 10))
    eta_min, eta_max = priors.get("eta", (0, 1))
    ach_min, ach_max = priors.get("ach", (0.3, 5))
    alpha_min, alpha_max = priors.get("alpha", (0, y.max() - y.min()))
    beta_min, beta_max = priors.get("beta", (0, y.min()))

    # initial values
    pos = np.array(
        [
            (u_value_min + u_value_max) / 2,  # u_value
            (T_sp_coo_min + T_sp_coo_max) / 2,  # T_sp_coo
            (T_sp_hea_min + T_sp_hea_max) / 2,  # T_sp_hea
            (T_sb_coo_min + T_sb_coo_max) / 2,  # T_sb_coo
            (T_sb_hea_min + T_sb_hea_max) / 2,  # T_sb_hea
            (eta_min + eta_max) / 2,  # eta
            (ach_min + ach_max) / 2,  # ach
            (alpha_min + alpha_max) / 2,  # alpha
            (beta_min + beta_max) / 2,  # beta
        ]
    )
    pos = np.tile(pos, (nwalkers, 1)) + np.random.randn(nwalkers, ndim)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sampler.run_mcmc(pos, n_samples, progress=verbose)

    chain = sampler.get_chain(discard=100, thin=15, flat=True)
    return chain


###############################################################################################
# Predictors
###############################################################################################


class UsagePredictor(RegressorMixin, BaseEstimator):
    """Predict energy consumption using Gradient Boosting regression.

    Args:
        lags (dict, optional): Dictionary with keys that correspond to feature names,
            and values that are lists containing the time lags of that feature to add
            as additional regressors. Defaults to None.
        cat_features (str or list of str, optional): The names of the categorical features
            in the input data. Defaults to None.
        validation_size (float, optional): The size of the validation dataset (as
            percentage of all training data) to use for identifying optimal number
            of iterations for the Gradient Boosting model. Defaults to 0.2.
        skip_calendar (bool, optional): If True, the model will not add calendar
            features to the input data. Defaults to False.
    """

    def __init__(
        self,
        lags: dict = None,
        cat_features: Union[str, List[str]] = None,
        validation_size: float = 0.2,
        skip_calendar: bool = False,
    ):
        self.lags = lags
        self.cat_features = cat_features
        self.validation_size = validation_size
        self.skip_calendar = skip_calendar

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the estimator with the available data.

        Args:
            X (pandas.DataFrame): Feature data.
            y (pandas.DataFrame): Target data.

        Returns:
            UsagePredictor: The fitted predictor.
        """

        self.has_neg_values_ = np.sum(y < 0).item() > 0
        cat_features = as_list(self.cat_features)

        if cat_features:
            self.cat_features_ = [name for name in cat_features if name in X.columns]
        else:
            self.cat_features_ = [
                name for name in get_categorical_cols(X, int_is_categorical=False)
            ]

        # Encode categorical features
        if self.cat_features_:
            self.ordinal_enc_ = SafeOrdinalEncoder(
                features=self.cat_features_, remainder="passthrough"
            )
            X = self.ordinal_enc_.fit_transform(X)
            X[self.cat_features_] = X[self.cat_features_].astype("category")

        if self.lags:
            X = _create_lagged(X, self.lags)

        if not self.skip_calendar:
            X = pd.concat([X, _create_calendar(X)], axis=1)

        model, scores = train_boost(
            X,
            y,
            cat_features=self.cat_features_,
            validation_size=self.validation_size,
            return_scores=True,
        )
        self.model_ = model
        self.scores_ = scores
        self.features_ = X.columns
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict energy consumption given input data.

        Args:
            X (pandas.DataFrame): Input data.

        Returns:
            pandas.Series: The prediction.
        """
        check_is_fitted(self, "fitted_")

        # Encode categorical features
        if self.cat_features_:
            X = self.ordinal_enc_.transform(X)
            X[self.cat_features_] = X[self.cat_features_].astype("category")

        if self.lags:
            X = _create_lagged(X, self.lags)

        if not self.skip_calendar:
            X = pd.concat([X, _create_calendar(X)], axis=1)

        if not X.columns.equals(self.features_):
            raise ValueError(
                f"Columns to predict {X.columns} differ from "
                f"ones from training {self.features_}"
            )

        y_pred = pd.Series(self.model_.predict(X), index=X.index)

        if not self.has_neg_values_:
            y_pred = y_pred.clip(lower=0)

        return y_pred


def explain_predicition(
    X: pd.DataFrame,
    y: pd.Series,
    A: float,
    V: float,
    nwalkers: int = 100,
    n_samples: int = 1000,
    priors: dict = {},
    verbose: bool = False,
):
    """Explain a prediction by estimating first-principle parameters of the underlying building.

    Args:
        X (pandas.DataFrame): Feature data.
        y (pandas.Series): The prediction time series.
        A (float): Area of building exposed to outdoors (m.2).
        V (float): Air volume of building (m.3)
        nwalkers (int, optional): The number of walkers in the ensemble. Defaults to 100.
        n_samples (int, optional): The number of samples to collect. Defaults to 1000.
        priors (dict, optional): The priors for the parameters to be estimated. It should be
            a dictionary with keys: "u_value", "T_sp_coo", "T_sp_hea", "T_sb_coo", "T_sb_hea",
            "eta", "ach", "alpha", "beta", and values that are tuples of minimum and maximum
            expected values. Defaults to {}.
        verbose (bool, optional): If True, a progress bar will be visible. Defaults to False.

    Returns:
        namedtuple: A namedtuple with keys "cond_load_on", "ven_load", "cond_load_off", "plug_load".
    """

    run_params = {}

    run_params["nwalkers"] = nwalkers
    run_params["n_samples"] = n_samples
    run_params["priors"] = priors
    run_params["verbose"] = verbose

    temperature = X["temperature"]
    activity = X["activity"]

    samples = _run_mcmc(
        temperature, activity, y, A=A, V=V, cp=1.005, rho=1.2, run_params=run_params
    )
    theta = samples.mean(axis=0)

    cond_load_on, ven_load, cond_load_off, plug_load = _forward_model(
        theta, temperature, activity, A, V, 1.005, 1.2
    )

    Bunch = namedtuple(
        "Bunch", ["cond_load_on", "ven_load", "cond_load_off", "plug_load"]
    )
    return Bunch(
        cond_load_on=cond_load_on,
        ven_load=ven_load,
        cond_load_off=cond_load_off,
        plug_load=plug_load,
    )
