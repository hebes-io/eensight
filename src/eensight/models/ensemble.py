# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from joblib import Memory
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from eensight.utils import check_X, check_y

from .linear import LinearPredictor

#########################################################################################
# Utilities
#########################################################################################


def calendar_encoding(X):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("The input is expected in pandas DataFrame form")
    if "dayofyear" in X.columns:
        return np.concatenate(
            (
                np.sin(2 * np.pi * X["dayofyear"] / 365.25).values.reshape(-1, 1),
                np.cos(2 * np.pi * X["dayofyear"] / 365.25).values.reshape(-1, 1),
            ),
            axis=1,
        )
    else:
        return np.concatenate(
            (
                np.sin(2 * np.pi * X.index.dayofyear / 365.25).values.reshape(-1, 1),
                np.cos(2 * np.pi * X.index.dayofyear / 365.25).values.reshape(-1, 1),
            ),
            axis=1,
        )


def calendar_distance(u, v):
    return np.linalg.norm(u - v)


#########################################################################################
# Ensemble models
#########################################################################################


class ParetoEnsemble(BaseEnsemble):
    def __init__(
        self,
        *,
        base_estimator: BaseEstimator,
        ensemble_parameters: List[Dict],
    ):
        self.ensemble_parameters = ensemble_parameters
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=len(ensemble_parameters),
        )

    @property
    def n_parameters(self):
        try:
            self.estimators_
        except AttributeError as exc:
            raise ValueError(
                "The number of parameters is acceccible only after "
                "the model has been fitted"
            ) from exc

        try:
            self.estimators_[0].n_parameters
        except AttributeError as exc:
            raise ValueError(
                "Cannot propagate `n_parameters` from base estimators"
            ) from exc

        n_parameters = 0
        for estimator in self.estimators_:
            n_parameters += estimator.n_parameters
        return math.ceil(n_parameters / self.n_estimators)

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        self._validate_estimator()
        self.estimators_ = []

        for params in self.ensemble_parameters:
            estimator = self._make_estimator(append=True)
            estimator.apply_optimal_params(**params)
            estimator.fit(X, y)

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, include_components=False):
        check_is_fitted(self, "fitted_")
        prediction = None

        for estimator in self.estimators_:
            pred = estimator.predict(
                X,
                include_components=include_components,
            )
            if prediction is None:
                prediction = pred
            else:
                prediction += pred

        prediction = prediction / self.n_estimators
        return prediction


class ABCDistanceEnsemble(BaseEnsemble, metaclass=ABCMeta):
    """Abstract class for ensemble prediction using a distance metric.

    Args:
        model_structure : dict
            The base model's configuration
        distance_metric : callable
            The metric function used for computing the distance between two observations. It must
            satisfy the following properties:
            Non-negativity: d(x, y) >= 0
            Identity: d(x, y) = 0 if and only if x == y
            Symmetry: d(x, y) = d(y, x)
            Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)
        n_estimators : int (default=1)
            The number of estimators.
        sigma : float (default=0.5)
            It controls the kernel width that generates weights for sampling the dataset for
            each estimator. Generally, only values between 0.1 and 2 make practical sense.
        metric_transformer : A sklearn-compatible transformer (default=None)
            The transformer is used for transforming the input `X` into a form that is understood
            by the distance metric.
        weight_method : str, default='softmin'
            Defines the way individual predictions from the estimators are weighted so that to
            generate the final prediction. It can be 'softmin' or 'argmin'.
        cache_location : str, pathlib.Path or None
            The path to use as a data store for the metric calculations. If None is given, no
            caching of the calculations is done.
        alpha : float (default=0.1)
            Regularization strength of the underlying ridge regression; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of
            the estimates. Larger values specify stronger regularization.
        fit_intercept : bool (default=False)
            Whether to fit the intercept for this model. If set to false, no intercept will be used
            in calculations.
    """

    def __init__(
        self,
        *,
        model_structure,
        distance_metric,
        n_estimators=1,
        sigma=0.5,
        metric_transformer=None,
        weight_method="softmin",
        cache_location=None,
        alpha=0.1,
        fit_intercept=False,
    ):
        if weight_method not in ("softmin", "argmin"):
            raise ValueError("`weight_method` must be either `softmin` or `argmin`")

        self.model_structure = model_structure
        self.distance_metric = distance_metric
        self.sigma = sigma
        self.metric_transformer = metric_transformer
        self.weight_method = weight_method
        self.cache_location = cache_location
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        super().__init__(
            base_estimator=LinearPredictor(model_structure=model_structure),
            n_estimators=n_estimators,
            estimator_params=("alpha", "fit_intercept"),
        )

        self.metric_ = Memory(cache_location, verbose=0).cache(distance_metric)
        

    @abstractmethod
    def _get_anchors(self, X):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of `ABCDistanceEnsemble`"
            " and it must implement the `_get_anchors` method"
        )

    @property
    def n_parameters(self):
        try:
            self.n_parameters_
        except AttributeError as exc:
            raise ValueError(
                "The number of parameters is acceccible only after "
                "the model has been used for prediction"
            ) from exc
        else:
            return self.n_parameters_

    def _fit(self, X, y):
        self._validate_estimator()
        self.estimators_ = []

        X = check_X(X)
        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]

        self.anchors_ = np.array(self._get_anchors(X))
        X_for_metric = np.array(self.metric_transformer.fit_transform(X))

        aggregators = []
        for anchor in self.anchors_:
            distances = pairwise_distances(
                X_for_metric, np.atleast_2d(anchor), metric=self.metric_
            )
            distances = distances.squeeze()
            sample_weight = np.exp(
                -(distances ** 2) / (2 * (self.sigma * distances.std()) ** 2)
            )

            mask = np.random.uniform(size=len(X)) <= sample_weight
            X_train, y_train = X[mask], y[mask]
            estimator = self._make_estimator()
            estimator.fit(X_train, y_train)

            pred = estimator.predict(X)
            mae = np.abs(y - pred)
            aggregators.append(
                IsotonicRegression(out_of_bounds="clip").fit(
                    distances, mae.values.squeeze()
                )
            )
        self.aggregators_ = aggregators

    def _calculate_n_parameters(self, weights):
        n_parameters = []
        for est in self.estimators_:
            n_parameters.append(est.n_parameters)
        n_parameters = np.mean(
            np.sum(weights * np.array(n_parameters).reshape(1, -1), axis=1)
        )
        self.n_parameters_ = math.ceil(n_parameters)

    def _predict(self, X: pd.DataFrame, include_components=False):
        X = check_X(X)
        X_for_metric = np.array(self.metric_transformer.fit_transform(X))
        distances = pairwise_distances(
            X_for_metric, np.atleast_2d(self.anchors_), metric=self.metric_
        )

        weights = np.zeros_like(distances)
        for i, agg in enumerate(self.aggregators_):
            weights[:, i] = agg.predict(distances[:, i])

        if self.weight_method == "softmin":
            weights = softmax(-weights, axis=1)
        elif self.weight_method == "argmin":
            closest = weights.min(axis=1).reshape(-1, 1)
            weights = np.where(weights == closest, 1, 0)

        self._calculate_n_parameters(weights)
        weights = pd.DataFrame(data=weights, index=X.index)
        prediction = None

        for i, estimator in enumerate(self.estimators_):
            pred = estimator.predict(X, include_components=include_components)
            pred = pred.mul(weights.iloc[:, i], axis=0)
            if prediction is None:
                prediction = pred
            else:
                prediction += pred
        return prediction


class CalendarEnsemble(ABCDistanceEnsemble):
    """A TOWT-like model using ensemble prediction.

    Args:
        model_structure : dict
            The base model's configuration
        n_estimators : int (default=1)
            The number of estimators.
        sigma : float (default=0.5)
            It controls the kernel width that generates weights for sampling the dataset for
            each estimator. Generally, only values between 0.1 and 2 make practical sense.
        weight_method : str, default='softmin'
            Defines the way individual predictions from the estimators are weighted so that to
            generate the final prediction. It can be 'softmin' or 'argmin'.
        cache_location : str, pathlib.Path or None
            The path to use as a data store for the metric calculations. If None is given, no
            caching of the calculations is done.
        alpha : float (default=0.1)
            Regularization strength of the underlying ridge regression; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of
            the estimates. Larger values specify stronger regularization.
        fit_intercept : bool (default=False)
            Whether to fit the intercept for this model. If set to false, no intercept will be used
            in calculations.
    """

    def __init__(
        self,
        *,
        model_structure,
        n_estimators=1,
        sigma=0.5,
        weight_method="softmin",
        cache_location=None,
        alpha=0.1,
        fit_intercept=False,
    ):
        super().__init__(
            model_structure=model_structure,
            distance_metric=calendar_distance,
            n_estimators=n_estimators,
            sigma=sigma,
            metric_transformer=FunctionTransformer(func=calendar_encoding),
            weight_method=weight_method,
            cache_location=cache_location,
            alpha=alpha,
            fit_intercept=fit_intercept,
        )

    def _get_anchors(self, X: pd.DataFrame):
        anchors = []
        X = X.sort_index()
        start = X.index.min()
        end = X.index.max()
        diff = (end - start) / (2 * self.n_estimators)

        for i in range(1, 2 * self.n_estimators, 2):
            anchors.append(start + diff * i)

        anchors = pd.DataFrame(index=anchors, dtype="float")
        return self.metric_transformer.fit_transform(anchors)

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. " "Instantiate a new object."
            )
        self._fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, include_components=False):
        check_is_fitted(self, "fitted_")
        return self._predict(X, include_components=include_components)

    @staticmethod
    def optimization_space(trial, **kwargs):
        n_estimators_lower = kwargs.get("n_estimators_lower") or 2
        n_estimators_upper = kwargs.get("n_estimators_upper") or 12
        sigma_lower = kwargs.get("sigma_lower") or 0.1
        sigma_upper = kwargs.get("sigma_upper") or 2.0

        param_space = {
            "n_estimators": trial.suggest_int(
                "n_estimators", n_estimators_lower, n_estimators_upper
            ),
            "sigma": trial.suggest_float("sigma", sigma_lower, sigma_upper),
        }
        return param_space

    def apply_optimal_params(self, **param_space):
        param_space.update(n_estimators=int(param_space["n_estimators"]))
        self.set_params(**param_space)
