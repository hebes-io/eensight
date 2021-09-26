# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/donlnz/nonconformist

from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils import Bunch
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted

from eensight.features import DatetimeFeatures
from eensight.utils import as_list, check_X, check_y, maybe_reshape_2d


class IcpEstimator(BaseEstimator):
    """Inductive conformal estimator.

    Args:
        estimator : Any regressor with scikit-learn predictor API (i.e. with fit and
            predict methods)
            The object to use to calculate the non-conformity scores. The estimator is
            expected to be the result of a cross-validation process and it must be already
            fitted.
        oos_mask : array-like
            The index of the training dataset's subset that the `estimator` has not seen
            during its fitting (i.e. the test sample of the relevant cross-validation fold).
        add_normalizer : bool, default=True
            If True, a normalization model will be added. Its predictions act as a
            multiplicative correction factor of the non-conformity scores. The
            normalization model is a random forest regressor
            (`sklearn.ensemble.RandomForestRegressor`).
        extra_regressors : str or list of str, default=None
            The names of the additional regressors to use for the normalization model. By
            default, the normalization model uses only the month of year, day of week, and
            hour of day features.
        n_estimators : int, default=100
            The number of trees in the normalization model.
        min_samples_leaf : int or float, default=0.05
            The minimum number of samples required to be at a leaf node of the
            normalization model. A split point at any depth will only be considered
            if it leaves at least ``min_samples_leaf`` training samples in each of
            the left and right branches.  This may have the effect of smoothing the
            model, especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.
        max_samples : int or float, default=0.8
            The number of samples to draw from X to train each base estimator in the
            normalization model.
            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
            `max_samples` should be in the interval `(0, 1)`.
    """

    def __init__(
        self,
        *,
        estimator: BaseEstimator,
        oos_mask: ArrayLike,
        add_normalizer=True,
        extra_regressors=None,
        n_estimators=100,
        min_samples_leaf=0.05,
        max_samples=0.8,
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
            features = features[as_list(self.extra_regressors)]
        return features

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            y : pd.DataFrame, shape (n_samples, 1)
                The target dataframe.

        Returns:
            self : object
                Returns self.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. " "Instantiate a new object."
            )

        X = check_X(X, exists=self.extra_regressors)
        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]

        X_oos = X[np.isin(X.index, self.oos_mask)]
        y_oos = y[np.isin(y.index, self.oos_mask)]
        prediction = self.estimator.predict(X_oos)

        if self.add_normalizer:
            self.normalizer_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_samples=self.max_samples,
            )
            residual_error = np.abs(y_oos - prediction)
            residual_error = residual_error.dropna()
            log_err = np.log1p(residual_error)
            features = self._generate_normalizer_features(X_oos)
            self.normalizer_.fit(np.array(features), np.array(log_err).squeeze())

            norm = pd.DataFrame(
                data=np.expm1(self.normalizer_.predict(np.array(features))),
                index=X_oos.index,
                columns=[self.target_name_],
            )
            nc_scores = np.abs(y_oos - prediction).divide(norm, axis=0)
        else:
            nc_scores = np.abs(y_oos - prediction)

        nc_scores = np.array(nc_scores.dropna()).squeeze()  # non-conformity scores
        self.nc_scores_ = nc_scores
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, significance=None):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            significance : float or list of floats between 0 and 1
                Significance level (maximum allowed error rate) of predictions. If ``None``,
                then intervals for all significance levels (0.01, 0.02, ..., 0.99) will be
                computed.

        Returns:
            result : sklearn.utils.Bunch
                A dict-like object with fields:
                - significance: The significance levels used in the calculations, list of float.
                - quantiles: The quantiles of the non-conformity scores, array of shape
                  (len(X), len(significance))
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

        return Bunch(significance=significance, quantiles=quantiles)


class AggregatedCp(BaseEstimator):
    """Aggregated conformal estimator. Combines multiple IcpRegressor estimators
    into an aggregated model.

    Args:
        estimators : List of estimators with scikit-learn predictor API (with fit
            and predict methods).
            Each estimator is expected to be the result of a cross-validation process
            and it must be already fitted.
        oos_masks : list of array-like
            List containing the index of the training dataset's subset that each `estimator`
            in `estimators` has not seen during its fitting (i.e. the test sample of the
            relevant cross-validation fold).
        add_normalizer : bool, default=True
            If True, a normalization model will be added. Its predictions act as a
            multiplicative correction factor of the non-conformity scores. The
            normalization model is a random forest regressor
            (`sklearn.ensemble.RandomForestRegressor`).
        extra_regressors : str or list of str, default=None
            The names of the additional regressors to use for the normalization model. By
            default, the normalization model uses only the month of year, day of week, and
            hour of day features.
        n_estimators : int, default=100
            The number of trees in the normalization model.
        min_samples_leaf : int or float, default=0.05
            The minimum number of samples required to be at a leaf node of the
            normalization model. A split point at any depth will only be considered
            if it leaves at least ``min_samples_leaf`` training samples in each of
            the left and right branches.  This may have the effect of smoothing the
            model, especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.
        max_samples : int or float, default=0.8
            The number of samples to draw from X to train each base estimator in the
            normalization model.
            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
            `max_samples` should be in the interval `(0, 1)`.
        n_jobs : int, default=None
            Number of jobs to run in parallel. ``None`` means 1 unless in a `joblib.parallel_backend`
            context. ``-1`` means using all processors.
    """

    def __init__(
        self,
        *,
        estimators: List[BaseEstimator],
        oos_masks: List[ArrayLike],
        add_normalizer=True,
        extra_regressors=None,
        n_estimators=100,
        min_samples_leaf=0.05,
        max_samples=0.8,
        n_jobs=None,
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
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            y : pd.DataFrame, shape (n_samples, 1)
                The target dataframe.

        Returns:
            self : object
                Returns self.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. " "Instantiate a new object."
            )

        parallel = Parallel(n_jobs=self.n_jobs)
        self.icp_estimators_ = parallel(
            delayed(self._fit_icp_estimator)(X, y, estimator, oos_mask)
            for estimator, oos_mask in zip(self.estimators, self.oos_masks)
        )
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, significance=None):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            significance : float or list of floats between 0 and 1
                Significance level (maximum allowed error rate) of predictions. If ``None``,
                then intervals for all significance levels (0.01, 0.02, ..., 0.99) will be
                computed.

        Returns:
            result : sklearn.utils.Bunch
                A dict-like object with fields:
                - significance: The significance levels used in the calculations, list of float.
                - quantiles: The quantiles of the non-conformity scores, array of shape
                  (len(X), len(significance))
        """
        check_is_fitted(self, "fitted_")

        if significance is None:
            significance = np.arange(0.01, 1.0, 0.01)
        else:
            significance = as_list(significance)

        n_obs = X.shape[0]
        quantiles = np.zeros((n_obs, len(significance)))

        parallel = Parallel(n_jobs=self.n_jobs)
        results = parallel(
            delayed(icp.predict)(X, significance=significance)
            for icp in self.icp_estimators_
        )

        for i, result in enumerate(results):
            quantiles += result.quantiles

        return Bunch(
            significance=significance,
            quantiles=quantiles / (i + 1),
        )
