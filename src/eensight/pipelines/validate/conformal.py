# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/donlnz/nonconformist

from functools import reduce
from typing import List, Union

import numpy as np
import pandas as pd
from feature_encoders.generate import DatetimeFeatures
from feature_encoders.utils import as_list, check_X, check_y, maybe_reshape_2d
from joblib import Parallel
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted


class IcpEstimator(BaseEstimator):
    """Inductive conformal estimator.

    Args:
        estimator (BaseEstimator): Any regressor with scikit-learn predictor API
            (i.e. with a `predict` method). The object to use to calculate the
            non-conformity scores. The estimator is expected to be the result of
            a cross-validation process and it must be already fitted.
        oos_mask (np.ndarray): The index of the training dataset's subset that the
            `estimator` has not seen during its fitting (i.e. the test sample of the
            relevant cross-validation fold).
        add_normalizer (bool, optional): If True, a normalization model will be added.
            Its predictions act as a multiplicative correction factor of the non-conformity
            scores. The normalization model is a `sklearn.ensemble.RandomForestRegressor`.
            Defaults to True.
        extra_regressors (str or list of str, optional): The names of the additional
            regressors to use for the normalization model. By default, the normalization
            model uses only the month of year, day of week, and hour of day features.
            Defaults to None.
        n_estimators (int, optional): The number of trees in the normalization model.
            Defaults to 100.
        min_samples_leaf (float, optional): The minimum number of samples required to be
            at a leaf node of the normalization model. A split point at any depth will only
            be considered if it leaves at least `ceil(min_samples_leaf * n_samples)` training
            samples in each of the left and right branches. Defaults to 0.05.
        max_samples (float, optional): The number of samples to draw from `X` to train each
            base estimator in the normalization model. `max_samples` should be in the interval
            `(0, 1)` since `max_samples * X.shape[0]` samples will be drawn. Defaults to 0.8.

    Raises:
        ValueError: if the `estimator` is not already fitted.
    """

    def __init__(
        self,
        *,
        estimator: BaseEstimator,
        oos_mask: np.ndarray,
        add_normalizer: bool = True,
        extra_regressors: Union[str, List[str]] = None,
        n_estimators: int = 100,
        min_samples_leaf: float = 0.05,
        max_samples: float = 0.8,
    ):
        try:
            check_is_fitted(estimator, "fitted_")
        except NotFittedError as exc:
            raise ValueError("The `estimator` must be already fitted") from exc

        self.estimator = estimator
        self.oos_mask = oos_mask
        self.add_normalizer = add_normalizer
        self.extra_regressors = extra_regressors
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples

    def _generate_normalizer_features(self, X):
        if self.extra_regressors is None:
            features = DatetimeFeatures(
                remainder="drop", subset=["hour", "dayofweek", "month"]
            ).fit_transform(X)
        else:
            features = DatetimeFeatures(
                remainder="passthrough", subset=["hour", "dayofweek", "month"]
            ).fit_transform(X)
            features = features[
                ["hour", "dayofweek", "month"] + as_list(self.extra_regressors)
            ]
        return features

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the estimator on the available data.

        Args:
            X (pandas.DataFrame): The input dataframe.
            y (pandas.DataFrame): The target dataframe.

        Returns:
            IcpEstimator: Fitted estimator.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        X = check_X(X, exists=self.extra_regressors)
        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]

        X_oos = X[np.isin(X.index, self.oos_mask)]
        y_oos = y[np.isin(y.index, self.oos_mask)]
        prediction = self.estimator.predict(X_oos)
        nc_scores = np.abs(y_oos - prediction)  # non-conformity scores

        if self.add_normalizer:
            self.normalizer_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_samples=self.max_samples,
            )
            log_err = np.log1p(nc_scores)
            features = self._generate_normalizer_features(X_oos)
            self.normalizer_.fit(np.array(features), np.array(log_err).squeeze())

            norm = pd.DataFrame(
                data=np.expm1(self.normalizer_.predict(np.array(features))),
                index=X_oos.index,
                columns=[self.target_name_],
            )
            nc_scores = nc_scores.divide(norm, axis=0)

        self.nc_scores_ = np.array(nc_scores).squeeze()
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, significance: Union[float, List[float]] = None):
        """Predict with uncertainty intervals.

        Args:
            X (pd.DataFrame): The input dataframe.
            significance (float or list of floats between 0 and 1, optional): Significance
                level (maximum allowed error rate) of predictions. If ``None``, then
                intervals for all significance levels (0.01, 0.02, ..., 0.99) will be
                computed. Defaults to None.

        Returns:
            pandas.DataFrame: A dataframe of shape `(len(X), len(significance))` with columns
                containing the significance levels used in the calculations, and data containing
                the quantiles of the non-conformity scores.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.extra_regressors)

        if significance is None:
            significance = np.arange(0.01, 1.0, 0.01)
        else:
            significance = as_list(significance)

        if self.add_normalizer:
            features = self._generate_normalizer_features(X)
            norm = np.expm1(self.normalizer_.predict(np.array(features)))

        n_obs = X.shape[0]
        quantiles = np.zeros((n_obs, len(significance)))

        for i, s in enumerate(significance):
            q = np.full(
                (n_obs, 1), np.quantile(self.nc_scores_, s, interpolation="higher")
            )
            if self.add_normalizer:
                q *= maybe_reshape_2d(np.array(norm))
            quantiles[:, i] = q[:, 0]

        return pd.DataFrame(data=quantiles, index=X.index, columns=significance)


class AggregatedCp(BaseEstimator):
    """Aggregated conformal estimator.

    Combines multiple IcpRegressor estimators into an aggregated model.

    Args:
        estimators (list of BaseEstimator): List of estimators with a `predict` methods.
            Each estimator is expected to be the result of a cross-validation process
            and it must be already fitted.
        oos_masks (numpy.ndarray): List containing the index of the training dataset's
            subset that each `estimator` in `estimators` has not seen during its fitting
            (i.e. the test sample of the relevant cross-validation fold).
        add_normalizer (bool, optional): If True, a normalization model will be added.
            Its predictions act as a multiplicative correction factor of the non-conformity
            scores. The normalization model is a `sklearn.ensemble.RandomForestRegressor`.
            Defaults to True.
        extra_regressors (str or list of str, optional): The names of the additional
            regressors to use for the normalization model. By default, the normalization
            model uses only the month of year, day of week, and hour of day features.
            Defaults to None.
        n_estimators (int, optional): The number of trees in the normalization model.
            Defaults to 100.
        min_samples_leaf (float, optional): The minimum number of samples required to be
            at a leaf node of the normalization model. A split point at any depth will only
            be considered if it leaves at least `ceil(min_samples_leaf * n_samples)` training
            samples in each of the left and right branches. Defaults to 0.05.
        max_samples (float, optional): The number of samples to draw from `X` to train each
            base estimator in the normalization model. `max_samples` should be in the interval
            `(0, 1)` since `max_samples * X.shape[0]` samples will be drawn. Defaults to 0.8.
        n_jobs (int, optional): Number of jobs to run in parallel. ``None`` means 1
            unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors. Defaults to None.
    """

    def __init__(
        self,
        *,
        estimators: List[BaseEstimator],
        oos_masks: List[np.ndarray],
        add_normalizer: bool = True,
        extra_regressors: Union[str, List[str]] = None,
        n_estimators: int = 100,
        min_samples_leaf: float = 0.05,
        max_samples: float = 0.8,
        n_jobs: int = None,
    ):
        self.estimators = estimators
        self.oos_masks = oos_masks
        self.add_normalizer = add_normalizer
        self.extra_regressors = extra_regressors
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.n_jobs = n_jobs

    def _fit_icp_estimator(self, X, y, estimator, oos_mask):
        icp = IcpEstimator(
            estimator=estimator,
            oos_mask=oos_mask,
            add_normalizer=self.add_normalizer,
            extra_regressors=self.extra_regressors,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_samples=self.max_samples,
        )
        return icp.fit(X, y)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the estimator on the available data.

        Args:
            X (pandas.DataFrame): The input dataframe.
            y (pandas.DataFrame): The target dataframe.

        Returns:
            AggregatedCp: Fitted estimator.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        parallel = Parallel(n_jobs=self.n_jobs)
        self.icp_estimators_ = parallel(
            delayed(self._fit_icp_estimator)(X, y, estimator, oos_mask)
            for estimator, oos_mask in zip(self.estimators, self.oos_masks)
        )
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, significance: Union[float, List[float]] = None):
        """Predict with uncertainty intervals.

        Args:
            X (pd.DataFrame): The input dataframe.
            significance (float or list of floats between 0 and 1, optional): Significance
                level (maximum allowed error rate) of predictions. If ``None``, then
                intervals for all significance levels (0.01, 0.02, ..., 0.99) will be
                computed. Defaults to None.

        Returns:
            pandas.DataFrame: A dataframe of shape `(len(X), len(significance))` with columns
                containing the significance levels used in the calculations, and data containing
                the quantiles of the non-conformity scores.
        """
        check_is_fitted(self, "fitted_")

        if significance is None:
            significance = np.arange(0.01, 1.0, 0.01)
        else:
            significance = as_list(significance)

        parallel = Parallel(n_jobs=self.n_jobs)
        results = parallel(
            delayed(icp.predict)(X, significance=significance)
            for icp in self.icp_estimators_
        )

        return reduce(lambda x, y: x.add(y), results) / len(results)
