# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from eensight.utils import as_list, check_X, get_datetime_data

#####################################################################################
# Add new features
# All feature generators generate pandas DataFrames
#####################################################################################


class TrendFeatures(TransformerMixin, BaseEstimator):
    """
    Generate time trend features

    Args:
        feature : str, default=None
            The name of the input dataframe's column that contains datetime information.
            If None, it is assumed that the datetime information is provided by the
            input dataframe's index.
        include_bias : bool, default=False
            If True, a column of ones is added to the output.
        remainder : str, :type : {'drop', 'passthrough'}, default='passthrough'
            By specifying ``remainder='passthrough'``, all the remaining columns of the
            input dataset will be automatically passed through (concatenated with the
            output of the transformer).
        replace : bool, default=False
            Specifies whether replacing an existing column with the same name is allowed
            (when `remainder=passthrough`).
    """

    def __init__(
        self, feature=None, include_bias=False, remainder="passthrough", replace=False
    ):
        if remainder not in ("passthrough", "drop"):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')

        self.feature = feature
        self.include_bias = include_bias
        self.remainder = remainder
        self.replace = replace

    def fit(self, X, y=None):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            y : None
                There is no need of a target in a transformer, but the pipeline
                API requires this parameter.

        Returns:
            self : object
                Returns self.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        X = check_X(X)
        dates = X.index.to_series() if self.feature is None else X[self.feature]
        self.t_scaler_ = MinMaxScaler().fit(dates.to_frame())
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : pd.DataFrame, shape (n_samples, n_features_out_)

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        dates = X.index.to_series() if self.feature is None else X[self.feature]

        if not self.include_bias:
            out = pd.DataFrame(
                data=self.t_scaler_.transform(dates.to_frame()),
                columns=["growth"],
                index=X.index,
            )
        else:
            out = pd.DataFrame(
                data=np.concatenate(
                    (np.ones((len(X), 1)), self.t_scaler_.transform(dates.to_frame())),
                    axis=1,
                ),
                columns=["offset", "growth"],
                index=X.index,
            )

        if self.remainder == "passthrough":
            common = list(set(X.columns) & set(out.columns))
            if common and not self.replace:
                raise ValueError(f"Found common column names {common}")
            elif common:
                X = X.drop(common, axis=1)
            out = pd.concat((X, out), axis=1)

        return out


class DatetimeFeatures(TransformerMixin, BaseEstimator):
    """
    Generate date and time features

    Args:
        feature : str, default=None
            The name of the input dataframe's column that contains datetime information.
            If None, it is assumed that the datetime information is provided by the
            input dataframe's index.
        remainder : str, :type : {'drop', 'passthrough'}, default='passthrough'
            By specifying ``remainder='passthrough'``, all the remaining columns of the
            input dataset will be automatically passed through (concatenated with the
            output of the transformer).
        replace : bool, default=False
            Specifies whether replacing an existing column with the same name is allowed
            (when `remainder=passthrough`).
        subset : str or list of str (default=None)
            The names of the features to generate. If None, all features will be produced:
            'month', 'week', 'dayofyear', 'dayofweek', 'hour', 'hourofweek'.
            The last 2 features are generated only if the timestep of the input's
            `feature` (or index if `feature` is None) is smaller than `pd.Timedelta(days=1)`.
    """

    def __init__(
        self, feature=None, remainder="passthrough", replace=False, subset=None
    ):
        if remainder not in ("passthrough", "drop"):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')

        self.feature = feature
        self.remainder = remainder
        self.replace = replace
        self.subset = subset

    def _get_all_attributes(self, dt_column):
        attr = ["month", "week", "dayofyear", "dayofweek"]

        dt = dt_column.diff()
        time_step = dt.iloc[dt.values.nonzero()[0]].min()
        if time_step < pd.Timedelta(days=1):
            attr = attr + ["hour", "hourofweek"]

        if self.subset is not None:
            attr = [i for i in attr if i in as_list(self.subset)]
        return attr

    def fit(self, X, y=None):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            y : None
                There is no need of a target in a transformer, but the pipeline
                API requires this parameter.

        Returns:
            self : object
                Returns self.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        X = check_X(X)
        dt_column = get_datetime_data(X, col_name=self.feature)
        self.attr_ = self._get_all_attributes(dt_column)
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : pd.DataFrame, shape (n_samples, n_features_out_)

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        dt_column = get_datetime_data(X, col_name=self.feature)

        out = {}
        for n in self.attr_:
            if n == "week":
                out[n] = (
                    dt_column.dt.isocalendar().week.astype(dt_column.dt.day.dtype)
                    if hasattr(dt_column.dt, "isocalendar")
                    else dt_column.dt.week
                )
            elif n == "hourofweek":
                out[n] = None
            else:
                out[n] = getattr(dt_column.dt, n)

        if "hourofweek" in out:
            out["hourofweek"] = 24 * out.get(
                "dayofweek", dt_column.dt.dayofweek
            ) + out.get("hour", dt_column.dt.hour)

        out = pd.DataFrame.from_dict(out)

        if self.remainder == "passthrough":
            common = list(set(X.columns) & set(out.columns))
            if common and not self.replace:
                raise ValueError(f"Found common column names {common}")
            elif common:
                X = X.drop(common, axis=1)
            out = pd.concat((X, out), axis=1)

        return out


class CyclicalFeatures(TransformerMixin, BaseEstimator):
    """Create cyclical (seasonal) features as fourier terms

    Args:
        seasonality : str
            The name of the seasonality.
        feature : str, default=None
            The name of the input dataframe's column that contains datetime information.
            If None, it is assumed that the datetime information is provided by the
            input dataframe's index.
        period : float, default=None
            Number of days in one period.
        fourier_order : int, default=None
            Number of Fourier components to use.
        remainder : str, :type : {'drop', 'passthrough'}, default='passthrough'
            By specifying ``remainder='passthrough'``, all the remaining columns of the
            input dataset will be automatically passed through (concatenated with the
            output of the transformer).
        replace : bool, default=False
            Specifies whether replacing an existing column with the same name is allowed
            (when `remainder=passthrough`).

    Note:
        The encoder can provide default values for `period` and `fourier_order` if `seasonality`
        is one of `daily`, `weekly` or `yearly`.
    """

    def __init__(
        self,
        *,
        seasonality,
        feature=None,
        period=None,
        fourier_order=None,
        remainder="passthrough",
        replace=False,
    ):
        if remainder not in ("passthrough", "drop"):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')

        self.seasonality = seasonality
        self.feature = feature
        self.period = period
        self.fourier_order = fourier_order
        self.remainder = remainder
        self.replace = replace

    @staticmethod
    def _fourier_series(dates, period, order):
        """Provides Fourier series components with the specified frequency
        and order.

        Args:
            dates: pd.Series containing timestamps.
            period: Number of days of the period.
            order: Number of components.

        Returns:
            Matrix with seasonality features.
        """
        # convert to days since epoch
        t = np.array(
            (dates - datetime(2000, 1, 1)).dt.total_seconds().astype(np.float64)
        ) / (3600 * 24.0)

        return np.column_stack(
            [
                fun((2.0 * (i + 1) * np.pi * t / period))
                for i in range(order)
                for fun in (np.sin, np.cos)
            ]
        )

    def fit(self, X, y=None):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            y : None
                There is no need of a target in a transformer, but the pipeline
                API requires this parameter.

        Returns:
            self : object
                Returns self.
        """
        if self.seasonality not in ["daily", "weekly", "yearly"]:
            if (self.period is None) or (self.fourier_order is None):
                raise ValueError(
                    "When adding custom seasonalities, values for "
                    "`period` and `fourier_order` must be specified."
                )
        if self.seasonality in ["daily", "weekly", "yearly"]:
            if self.period is None:
                self.period = (
                    1
                    if self.seasonality == "daily"
                    else 7
                    if self.seasonality == "weekly"
                    else 365.25
                )
            if self.fourier_order is None:
                self.fourier_order = (
                    4
                    if self.seasonality == "daily"
                    else 3
                    if self.seasonality == "weekly"
                    else 6
                )
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : pd.DataFrame, shape (n_samples, n_features_out_)

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        X = check_X(X)
        dt_column = get_datetime_data(X, col_name=self.feature)
        out = self._fourier_series(dt_column, self.period, self.fourier_order)
        out = pd.DataFrame(
            data=out,
            index=X.index,
            columns=[
                f"{self.seasonality}_delim_{i}" for i in range(2 * self.fourier_order)
            ],
        )

        if self.remainder == "passthrough":
            common = list(set(X.columns) & set(out.columns))
            if common and not self.replace:
                raise ValueError(f"Found common column names {common}")
            elif common:
                X = X.drop(common, axis=1)
            out = pd.concat((X, out), axis=1)

        return out


class MMCFeatures(TransformerMixin, BaseEstimator):
    """
    Generate `month` and `dayofweek` one-hot encoded features for training an MMC
    (Mahalanobis Metric for Clustering) metric learning algorithm.

    Args:
        month_feature : str, default='month'
            The name of the input dataframe's column that contains the `month` feature values.
        dow_feature : str, default='dayofweek'
            The name of the input dataframe's column that contains the `dayofweek` feature values.
    """

    def __init__(self, month_feature="month", dow_feature="dayofweek"):
        self.month_feature = month_feature
        self.dow_feature = dow_feature

    def fit(self, X, y=None):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            y : None
                There is no need of a target in a transformer, but the pipeline
                API requires this parameter.

        Returns:
            self : object
                Returns self.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        X = check_X(X, exists=[self.month_feature, self.dow_feature])
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : pd.DataFrame, shape (n_samples, n_features_out_)

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=[self.month_feature, self.dow_feature])
        features = pd.DataFrame(0, index=X.index, columns=range(1, 20))

        temp = pd.get_dummies(X[self.month_feature].astype(int))
        for col in temp.columns:
            features[col] = temp[col]

        temp = pd.get_dummies(X[self.dow_feature].astype(int))
        for col in temp.columns:
            features[col + 13] = temp[col]

        return features
