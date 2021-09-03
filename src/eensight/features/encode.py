# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype as is_bool
from pandas.api.types import is_categorical_dtype as is_category
from pandas.api.types import is_integer_dtype as is_integer
from pandas.api.types import is_object_dtype as is_object
from scipy.stats import skew, wasserstein_distance
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from eensight.features._polynomial import SplineTransformer
from eensight.utils import (
    add_constant,
    as_list,
    check_X,
    check_y,
    get_datetime_data,
    maybe_reshape_2d,
    tensor_product,
)

logger = logging.getLogger("feature-encoding")

UNKNOWN_VALUE = -1


#####################################################################################
# IdentityEncoder (utility encoder)
#####################################################################################


class IdentityEncoder(TransformerMixin, BaseEstimator):
    """
    The identity encoder returns what it is fed.

    Args:
        feature : str or list of str, default=None
            The name(s) of the input dataframe's column(s) to return.
            If None, the whole input dataframe is returned.
        include_bias : bool, default=False
            If True, a column of ones is added to the output.
    """

    def __init__(self, feature=None, include_bias=False):
        self.feature = feature
        self.include_bias = include_bias
        self.features_ = as_list(feature)

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
        n_features_out_ = len(self.features_) if self.features_ else X.shape[1]
        self.n_features_out_ = int(self.include_bias) + n_features_out_
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array

        Raises:
            ValueError: If `include_bias` is True and a column with constant
            values already exists in the returned columns, or if the input data
            do not pass the checks of `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X[self.features_]) if self.features_ else check_X(X)
        if self.include_bias:
            X = add_constant(X, has_constant="raise")

        return np.array(X)


#####################################################################################
# Time trend features
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
    """

    def __init__(self, feature=None, include_bias=False):
        self.feature = feature
        self.include_bias = include_bias

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
        self.n_features_out_ = 1 + int(self.include_bias)
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        dates = X.index.to_series() if self.feature is None else X[self.feature]

        if not self.include_bias:
            return self.t_scaler_.transform(dates.to_frame())
        else:
            return np.concatenate(
                (np.ones((len(X), 1)), self.t_scaler_.transform(dates.to_frame())),
                axis=1,
            )


#####################################################################################
# Date and time features
#####################################################################################


class DatetimeFeatures(TransformerMixin, BaseEstimator):
    """
    Generate date and time features

    Args:
        feature : str, default=None
            The name of the input dataframe's column that contains datetime information.
            If None, it is assumed that the datetime information is provided by the
            input dataframe's index.
        remainder : str, :type : {'drop', 'passthrough'}, default='drop'
            By specifying ``remainder='passthrough'``, all the remaining columns of the
            input dataset will be automatically passed through (concatenated with the
            output of the transformer).
        replace : bool, default=False
            Specifies whether replacing an existing column with the same name is allowed
            (when `remainder=passthrough`).
        subset : str or list of str (default=None)
            The names of the features to generate. If None, all features will be produced:
            'month', 'week', 'dayofyear', 'dayofweek', 'hour', 'hourofweek', 'time'.
            The last 3 features are generated only if the timestep of the input's
            `feature` (or index if `feature` is None) is smaller than `pd.Timedelta(days=1)`.
    """

    def __init__(self, feature=None, remainder="drop", replace=False, subset=None):
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
            attr = attr + ["hour", "hourofweek", "time"]

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
        n_features_out_ = len(self.attr_)
        self.n_features_out_ = (
            n_features_out_ + int(self.remainder == "passthrough") * X.shape[1]
        )
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

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
        feature : str, default=None
            The name of the input dataframe's column that contains datetime information.
            If None, it is assumed that the datetime information is provided by the
            input dataframe's index.
        seasonality : str, default=None
            The name of the seasonality.
        period : float
            Number of days in one period.
        fourier_order : int
            Number of Fourier components to use.

    Note:
        The encoder can provide default values for `period` and `fourier_order` if `seasonality`
        is one of `daily`, `weekly` or `yearly`.
    """

    def __init__(
        self,
        feature=None,
        seasonality=None,
        period=None,
        fourier_order=None
    ):
        self.feature = feature
        self.seasonality = seasonality
        self.period = period
        self.fourier_order = fourier_order

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
        self.n_features_out_ = 2 * self.fourier_order
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.
        
        Raises:
            ValueError: If the input data do not pass the checks of `eensight.utils.check_X`.
        """
        X = check_X(X)
        dt_column = get_datetime_data(X, col_name=self.feature)
        out = self._fourier_series(dt_column, self.period, self.fourier_order)
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
        self.n_features_out_ = 19
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

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


#############################################################################
# Encode categorical data
#############################################################################


class SafeOrdinalEncoder(TransformerMixin, BaseEstimator):
    """
    Encode categorical features as an integer array. The encoder converts the 
    features into ordinal integers. This results in a single column of integers 
    (0 to n_categories - 1) per feature.

    Args:
        feature : str or list of str, default=None
            The names of the columns to encode. If None, all categorical columns will
            be encoded.
        unknown_value : int, default=None
            This parameter will set the encoded value for unknown categories. It has to
            be distinct from the values used to encode any of the categories in `fit`.
            If None, the value `-1` is used. During `transform`, unknown categories
            will be replaced using the most frequent value along each column.
    """

    def __init__(self, feature=None, unknown_value=None):
        self.feature = feature
        self.unknown_value = unknown_value
        self.features_ = as_list(feature)

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
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        X, categorical_cols, _ = check_X(X, exists=self.features_, return_col_info=True)

        if not self.features_:
            self.features_ = categorical_cols
        else:
            for name in self.features_:
                if pd.api.types.is_float_dtype(X[name]):
                    raise ValueError(f"The encoder is applied on numerical data")

        self.feature_pipeline_ = Pipeline(
            [
                (
                    "select",
                    ColumnTransformer(
                        [("select", "passthrough", self.features_)], remainder="drop"
                    ),
                ),
                (
                    "encode_ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=self.unknown_value or UNKNOWN_VALUE,
                        dtype=np.int16,
                    ),
                ),
                (
                    "impute_unknown",
                    SimpleImputer(
                        missing_values=self.unknown_value or UNKNOWN_VALUE,
                        strategy="most_frequent",
                    ),
                ),
            ]
        )
        # Fit the pipeline
        self.feature_pipeline_.fit(X)
        self.n_features_out_ = len(self.features_)
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.features_)
        return self.feature_pipeline_.transform(X)


class SafeOneHotEncoder(TransformerMixin, BaseEstimator):
    """
    The encoder uses a `SafeOrdinalEncoder`to first encode the feature as an integer 
    array and then a `sklearn.preprocessing.OneHotEncoder` to encode the features as 
    an one-hot array. 

    Args:
        feature : str or list of str, default=None
            The names of the columns to encode. If None, all categorical columns will
            be encoded.
        unknown_value : int, default=None
            This parameter will set the encoded value of unknown categories. It has to
            be distinct from the values used to encode any of the categories in `fit`.
            If None, the value `-1` is used. During `transform`, unknown categories
            will be replaced using the most frequent value along each column.
    """

    def __init__(self, feature=None, unknown_value=None):
        self.feature = feature
        self.unknown_value = unknown_value
        self.features_ = as_list(feature)

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
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        X, categorical_cols, _ = check_X(X, exists=self.features_, return_col_info=True)

        if not self.features_:
            self.features_ = categorical_cols
        else:
            for name in self.features_:
                if pd.api.types.is_float_dtype(X[name]):
                    raise ValueError(f"The encoder is applied on numerical data")

        self.feature_pipeline_ = Pipeline(
            [
                (
                    "encode_ordinal",
                    SafeOrdinalEncoder(
                        feature=self.features_,
                        unknown_value=self.unknown_value or UNKNOWN_VALUE,
                    ),
                ),
                ("one_hot", OneHotEncoder(drop=None, sparse=False)),
            ]
        )
        # Fit the pipeline
        self.feature_pipeline_.fit(X)

        self.n_features_out_ = 0
        for category in self.feature_pipeline_["one_hot"].categories_:
            self.n_features_out_ += len(category)
        
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.features_)
        return self.feature_pipeline_.transform(X)


class TargetClusterEncoder(TransformerMixin, BaseEstimator):
    """Encodes a categorical feature as clusters of the target's values so that
    to reduce its cardinality.

    Args:
        feature : str
            The name of the categorical feature to transform. This encoder operates 
            on a single feature.
        max_n_categories : int
            The maximum number of categories to produce.
        stratify_by : str or list of str (default=None)
            If not None, the encoder will first stratify the categorical feature into
            groups that have similar values of the features in `stratify_by`, and then
            cluster based on the relationship between the categorical feature and the
            target.
        excluded_categories : str or list of str (default=None)
            The names of the categories to be excluded from the clustering process. These
            categories will stay intact by the encoding process, so they cannot have the
            same values as the encoder's results (the encoder acts as an ``OrdinalEncoder``
            in the sense that the feature is converted into a column of integers 0 to
            n_categories - 1).
        unknown_value : int, default=None
            This parameter will set the encoded value of unknown categories. It has to
            be distinct from the values used to encode any of the categories in `fit`.
            If None, the value `-1` is used.
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node of the decision
            tree model that is used for stratifying the categorical feature if `stratify_by`
            is not None. The actual number that will be passed to the tree model is
            `min_samples_leaf` multiplied by the number of unique values in the categorical
            feature to transform.
        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split of the decision tree.
            - If float, then `max_features` is a fraction and
            `int(max_features * n_features)` features are considered at each
            split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator. To obtain a deterministic behaviour
            during fitting, ``random_state`` has to be fixed to an integer.

    Note:
        This encoder does not replace unknown values with the most frequent one during
        `transform`. It just assigns them the value of `unknown_value`.
    """

    def __init__(
        self,
        *,
        feature,
        max_n_categories,
        stratify_by=None,
        excluded_categories=None,
        unknown_value=None,
        min_samples_leaf=1,
        max_features="auto",
        random_state=None,
    ):
        self.feature = feature
        self.max_n_categories = max_n_categories
        self.stratify_by = stratify_by
        self.excluded_categories = excluded_categories
        self.unknown_value = unknown_value
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.stratify_by_ = as_list(stratify_by)
        self.excluded_categories_ = as_list(excluded_categories)
        self.max_n_categories_ = self.max_n_categories - len(self.excluded_categories_)

        if self.max_n_categories_ < 2:
            raise ValueError(
                "The difference between `max_n_categories` and the number of "
                f"`excluded_categories` must be at least 2, but is: {self.max_n_categories_}."
            )

    def fit(self, X, y):
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
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        X = check_X(X, exists=[self.feature] + self.stratify_by_)
        if pd.api.types.is_float_dtype(X[self.feature]):
            raise ValueError(f"The encoder is applied on numerical data")

        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]

        X = X.merge(y, left_index=True, right_index=True)

        if self.excluded_categories_:
            unique_vals = X[self.feature].unique()
            for value in self.excluded_categories_:
                if value not in unique_vals:
                    raise ValueError(
                        f"Value {value} of `excluded_categories` not found "
                        f"in the {self.feature} data."
                    )

            mask = X[self.feature].isin(self.excluded_categories_)
            X = X.loc[~mask]

        if not self.stratify_by_:
            self.mapping_ = self._cluster_without_stratify(X)
        else:
            self.mapping_ = self._cluster_with_stratify(X)

        if self.excluded_categories_:
            start = len(self.mapping_)
            for i, cat in enumerate(self.excluded_categories_):
                self.mapping_.update({cat: start + i})

        self.n_features_out_ = 1
        self.fitted_ = True
        return self

    def _cluster_without_stratify(self, X):
        reference = np.array(X[self.target_name_])
        X = X.groupby(self.feature)[self.target_name_].agg(
            ["mean", "std", skew, lambda x: wasserstein_distance(x, reference)]
        )
        X.fillna(value=1, inplace=True)

        X_to_cluster = StandardScaler().fit_transform(X)
        n_clusters = min(X_to_cluster.shape[0], self.max_n_categories_)
        clusterer = KMeans(n_clusters=n_clusters)

        with warnings.catch_warnings(record=True) as warning:
            cluster_labels = pd.Series(
                data=clusterer.fit_predict(X_to_cluster), index=X.index
            )
            for w in warning:
                logger.warning(str(w))
        return cluster_labels.to_dict()

    def _cluster_with_stratify(self, X):
        X_train = None
        for col in self.stratify_by_:
            if (
                is_bool(X[col])
                or is_object(X[col])
                or is_category(X[col])
                or is_integer(X[col])
            ):
                X_train = pd.concat((X_train, pd.get_dummies(X[col])), axis=1)
            else:
                X_train = pd.concat((X_train, X[col]), axis=1)

        y_train = X[self.target_name_]
        n_categories = X[self.feature].nunique()

        min_samples_leaf = n_categories * int(self.min_samples_leaf)
        model = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        model = model.fit(X_train, y_train)

        leaf_ids = model.apply(X_train)
        uniq_ids = np.unique(leaf_ids)
        leaf_samples = [np.where(leaf_ids == id)[0] for id in uniq_ids]

        X_to_cluster = pd.DataFrame(
            index=X[self.feature].unique(), columns=range(len(leaf_samples))
        )
        for i, idx in enumerate(leaf_samples):
            subset = X.iloc[idx][[self.feature, self.target_name_]]
            a = subset.groupby(self.feature)[self.target_name_].mean()
            a = a.reindex(X_to_cluster.index)
            X_to_cluster.iloc[:, i] = a

        X_to_cluster = X_to_cluster.fillna(X_to_cluster.median())
        n_clusters = min(X_to_cluster.shape[0], self.max_n_categories_)
        clusterer = KMeans(n_clusters=n_clusters)

        with warnings.catch_warnings(record=True) as warning:
            cluster_labels = pd.Series(
                data=clusterer.fit_predict(X_to_cluster), index=X_to_cluster.index
            )
            for w in warning:
                logger.warning(str(w))
        return cluster_labels.to_dict()

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.feature)

        return maybe_reshape_2d(
            np.array(
                X[self.feature].map(
                    lambda x: int(
                        self.mapping_.get(x, self.unknown_value or UNKNOWN_VALUE)
                    )
                )
            )
        )


class CategoricalEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical features.

    Args:
        feature : str
            The name of the categorical feature to transform. This encoder operates on 
            a single feature.
        max_n_categories : int (default=None)
            The maximum number of categories to produce.
        stratify_by : str or list of str (default=None)
            If not None, the encoder will first stratify the categorical feature into
            groups that have similar values of the features in `stratify_by`, and then
            cluster based on the relationship between the categorical feature and the
            target.
        excluded_categories : str or list of str (default=None)
            The names of the categories to be excluded from the clustering process. These
            categories will stay intact by the encoding process, so they cannot have the
            same values as the encoder's results (the encoder acts as an ``OrdinalEncoder``
            in the sense that the feature is converted into a column of integers 0 to
            n_categories - 1).
        unknown_value : int, default=None
            This parameter will set the encoded value of unknown categories. It has to
            be distinct from the values used to encode any of the categories in `fit`.
            If None, the value `-1` is used.
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node of the decision
            tree model that is used for stratifying the categorical feature if `stratify_by`
            is not None. The actual number that will be passed to the tree model is
            `min_samples_leaf` multiplied by the number of unique values in the categorical
            feature to transform.
        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split of the decision tree.
            - If float, then `max_features` is a fraction and
            `int(max_features * n_features)` features are considered at each
            split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator. To obtain a deterministic behaviour
            during fitting, ``random_state`` has to be fixed to an integer.
        encode_as : str {'onehot', 'ordinal'}, default='onehot'
            Method used to encode the transformed result.
            onehot
                Encode the transformed result with one-hot encoding
                and return a dense array.
            ordinal
                Encode the transformed result as integer values.
    """

    def __init__(
        self,
        *,
        feature,
        max_n_categories=None,
        stratify_by=None,
        excluded_categories=None,
        unknown_value=None,
        min_samples_leaf=1,
        max_features="auto",
        random_state=None,
        encode_as="onehot",
    ):
        self.feature = feature
        self.max_n_categories = max_n_categories
        self.stratify_by = stratify_by
        self.excluded_categories = excluded_categories
        self.unknown_value = unknown_value
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.encode_as = encode_as
        self.excluded_categories_ = as_list(excluded_categories)

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
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        X = check_X(X, exists=self.feature)
        if pd.api.types.is_float_dtype(X[self.feature]):
            raise ValueError(f"The encoder is applied on numerical data")

        n_categories = X[self.feature].nunique()
        use_target = (self.max_n_categories is not None) and (
            n_categories > self.max_n_categories
        )

        if use_target and (y is None):
            raise ValueError(
                f"The number of categories {n_categories} is larger than "
                f"`max_n_categories`:{self.max_n_categories}. In this case, "
                "the target values must be provided for target-based encoding."
            )

        if not use_target:
            self.feature_pipeline_ = Pipeline(
                [
                    (
                        "encode_features",
                        SafeOneHotEncoder(
                            feature=self.feature, unknown_value=self.unknown_value
                        ),
                    )
                    if self.encode_as == "onehot"
                    else (
                        "encode_features",
                        SafeOrdinalEncoder(
                            feature=self.feature, unknown_value=self.unknown_value
                        ),
                    )
                ]
            )
        else:
            self.feature_pipeline_ = Pipeline(
                [
                    (
                        "reduce_dimension",
                        TargetClusterEncoder(
                            feature=self.feature,
                            stratify_by=self.stratify_by,
                            max_n_categories=self.max_n_categories,
                            excluded_categories=self.excluded_categories,
                            unknown_value=self.unknown_value,
                            min_samples_leaf=self.min_samples_leaf,
                            max_features=self.max_features,
                            random_state=self.random_state,
                        ),
                    ),
                    (
                        "to_pandas",
                        FunctionTransformer(
                            lambda x: pd.DataFrame(x, columns=[self.feature])
                        ),
                    ),
                    (
                        "encode_features",
                        SafeOneHotEncoder(
                            feature=self.feature, unknown_value=self.unknown_value
                        ),
                    )
                    if self.encode_as == "onehot"
                    else (
                        "encode_features",
                        SafeOrdinalEncoder(
                            feature=self.feature, unknown_value=self.unknown_value
                        ),
                    ),
                ]
            )

        # Fit the pipeline
        self.feature_pipeline_.fit(X, y)
        self.n_features_out_ = self.feature_pipeline_["encode_features"].n_features_out_
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.feature)
        return self.feature_pipeline_.transform(X)


#############################################################################
# Encoding pairwise categorical data interactions
#############################################################################


class ICatEncoder(TransformerMixin, BaseEstimator):
    """Encode the interaction between two categorical features

    Args:
        encoder_left : eensight.features.encode.CategoricalEncoder
            The encoder for the first of the two features
        encoder_right : eensight.features.encode.CategoricalEncoder
            The encoder for the second of the two features

    Note:
        Both encoders should have the same `encode_as` parameter.
        If one or both of the encoders is already fitted, it will not be
        re-fitted during `fit` or `fit_transform`.
    """

    def __init__(
        self, encoder_left: CategoricalEncoder, encoder_right: CategoricalEncoder
    ):
        if encoder_left.encode_as != encoder_right.encode_as:
            raise ValueError(
                "Both encoders should have the same `encode_as` parameter."
            )

        self.encoder_left = encoder_left
        self.encoder_right = encoder_right
        self.encode_as_ = encoder_left.encode_as

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
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        for encoder in (self.encoder_left, self.encoder_right):
            try:
                check_is_fitted(encoder, "fitted_")
            except NotFittedError:
                encoder.fit(X, y)

        self.n_features_out_ = (
            self.encoder_left.n_features_out_ * self.encoder_right.n_features_out_
        )
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")

        X_left = self.encoder_left.transform(X)
        X_right = self.encoder_right.transform(X)

        if self.encode_as_ == "onehot":
            return tensor_product(X_left, X_right)
        else:
            X_left = X_left.astype(str, copy=False)
            X_right = X_right.astype(str, copy=False)
            X_left = np.core.defchararray.add(X_left, np.array([":"]))
            return np.core.defchararray.add(X_left, X_right)


########################################################################################
# Encoding numerical data
########################################################################################


class SplineEncoder(TransformerMixin, BaseEstimator):
    """Generate univariate B-spline bases for features.
    Generates a new feature matrix consisting of `n_splines=n_knots + degree - 1`
    spline basis functions (B-splines) of polynomial order=`degree` for each
    feature.

    Args:
        feature : str
            The name of the column to encode.
        n_knots : int, default=5
            Number of knots of the splines if `knots` equals one of {'uniform', 'quantile'}.
            Must be larger or equal 2. Ignored if `knots` is array-like.
        degree : int, default=3
            The polynomial degree of the spline basis. Must be a non-negative integer.
        strategy : {'uniform', 'quantile'} or array-like of shape (n_knots, n_features),
            default='quantile'
            Set knot positions such that first knot <= features <= last knot.
            - If 'uniform', `n_knots` number of knots are distributed uniformly
            from min to max values of the features (each bin has the same width).
            - If 'quantile', they are distributed uniformly along the quantiles of
            the features (each bin has the same number of observations).
            - If an array-like is given, it directly specifies the sorted knot
            positions including the boundary knots. Note that, internally,
            `degree` number of knots are added before the first knot, the same
            after the last knot.
        extrapolation : {'error', 'constant', 'linear', 'continue'}, default='constant'
            If 'error', values outside the min and max values of the training features
            raises a `ValueError`. If 'constant', the value of the splines at minimum
            and maximum value of the features is used as constant extrapolation. If
            'linear', a linear extrapolation is used. If 'continue', the splines are
            extrapolated as is, i.e. option `extrapolate=True` in `scipy.interpolate.BSpline`.
        include_bias : bool, default=False
            If False, then the last spline element inside the data range of a feature
            is dropped. As B-splines sum to one over the spline basis functions for each
            data point, they implicitly include a bias term.
        order : {'C', 'F'}, default='C'
            Order of output array. 'F' order is faster to compute, but may slow
            down subsequent estimators.
    """

    def __init__(
        self,
        *,
        feature,
        n_knots=5,
        degree=3,
        strategy="quantile",
        extrapolation="constant",
        include_bias=False,
        order="C",
    ):
        self.feature = feature
        self.n_knots = n_knots
        self.degree = degree
        self.strategy = strategy
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order

    def fit(self, X, y=None, sample_weight=None):
        """Compute knot positions of splines.

        Args:
            X : pd.DataFrame of shape (n_samples, n_features)
                The data to fit.
            y : None
                Ignored.
            sample_weight : array-like of shape (n_samples,), default = None
                Individual weights for each sample. Used to calculate quantiles if
                `strategy="quantile"`. For `strategy="uniform"`, zero weighted
                observations are ignored for finding the min and max of `X`.

        Returns:
            self : object
                Fitted transformer.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        X = check_X(X, exists=self.feature)
        self.encoder_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots=self.strategy,
            extrapolation=self.extrapolation,
            include_bias=self.include_bias,
            order=self.order,
        )

        self.encoder_.fit(X[[self.feature]])
        self.n_features_out_ = self.encoder_.n_features_out_
        self.fitted_ = True
        return self

    def transform(self, X):
        """Transform each feature data to B-splines.

        Args:
            X : pd.DataFrame of shape (n_samples, n_features)
                The data to transform.

        Returns:
            XBS : ndarray of shape (n_samples, n_features_out_)
                The matrix of features, where n_splines is the number of bases
                elements of the B-splines, n_knots + degree - 1.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X, exists=self.feature)
        return self.encoder_.transform(X[[self.feature]])


#######################################################################################
# Encoding pairwise interactions between numerical features
#######################################################################################


class ISplineEncoder(TransformerMixin, BaseEstimator):
    """Encode the interaction between two spline-encoded numerical features

    Args:
        encoder_left : eensight.features.encode.SplineEncoder
            The encoder for the first of the two features
        encoder_right : eensight.features.encode.SplineEncoder
            The encoder for the second of the two features

    Note:
        If one or both of the encoders is already fitted, it will not be
        re-fitted during `fit` or `fit_transform`.
    """

    def __init__(self, encoder_left: SplineEncoder, encoder_right: SplineEncoder):
        if encoder_left.include_bias and encoder_right.include_bias:
            raise ValueError("`include_bias` cannot be True for both encoders")

        self.encoder_left = encoder_left
        self.encoder_right = encoder_right

    def fit(self, X, y=None):
        """
        Args:
            X : pd.DataFrame of shape (n_samples, n_features)
                The data to fit.
            y : None
                Ignored.

        Returns:
            self : object
                Fitted transformer.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        for encoder in (self.encoder_left, self.encoder_right):
            try:
                check_is_fitted(encoder, "fitted_")
            except NotFittedError:
                encoder.fit(X)

        self.n_features_out_ = (
            self.encoder_left.n_features_out_ * self.encoder_right.n_features_out_
        )
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.

        Raises:
            ValueError: If the input data do not pass the checks of
            `eensight.utils.check_X`.
        """
        check_is_fitted(self, "fitted_")

        X_left = self.encoder_left.transform(X)
        X_right = self.encoder_right.transform(X)
        return tensor_product(X_left, X_right)


class ProductEncoder(TransformerMixin, BaseEstimator):
    """Encode the interaction between two linear numerical features

    Args:
        encoder_left : eensight.features.encode.IdentityEncoder
            The encoder for the first of the two features
        encoder_right : eensight.features.encode.IdentityEncoder
            The encoder for the second of the two features

    Note:
        If one or both of the encoders is already fitted, it will not be
        re-fitted during `fit` or `fit_transform`.
    """

    def __init__(self, encoder_left: IdentityEncoder, encoder_right: IdentityEncoder):
        if encoder_left.include_bias or encoder_right.include_bias:
            raise ValueError("`include_bias` cannot be True for any of the encoders")

        if (len(encoder_left.features_) > 1) or (len(encoder_right.features_) > 1):
            raise ValueError("This encoder supports only pairwise interactions")

        self.encoder_left = encoder_left
        self.encoder_right = encoder_right

    def fit(self, X, y=None):
        """
        Args:
            X : pd.DataFrame of shape (n_samples, n_features)
                The data to fit.
            y : None
                Ignored.

        Returns:
            self : object
                Fitted transformer.
        """
        for encoder in (self.encoder_left, self.encoder_right):
            try:
                check_is_fitted(encoder, "fitted_")
            except NotFittedError:
                encoder.fit(X)

        self.n_features_out_ = 1
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.
        """
        check_is_fitted(self, "fitted_")
        X_left = self.encoder_left.transform(X)
        X_right = self.encoder_right.transform(X)
        return np.multiply(X_left, X_right)


###############################################################################
# Encoding pairwise interactions of one numerical and one categorical feature
###############################################################################


class ICatLinearEncoder(TransformerMixin, BaseEstimator):
    """Encode the interaction between one categorical and one linear numerical
    feature.

    Args:
        encoder_cat : eensight.features.encode.CategoricalEncoder
            The encoder for the categorical feature. It must encode features in
            an one-hot form.
        encoder_num : eensight.features.encode.IdentityEncoder
            The encoder for the numerical feature. The  encoder should have
            `include_bias=True`. This is necessary so that so that it is possible
            to model a different intercept for each categorical feature's level.

    Note:
        If the categorical encoder is already fitted, it will not be re-fitted during `fit`
        or `fit_transform`.
    """

    def __init__(
        self, *, encoder_cat: CategoricalEncoder, encoder_num: IdentityEncoder
    ):
        if encoder_cat.encode_as != "onehot":
            raise ValueError(
                "This encoder supports only one-hot encoding of the "
                "categorical feature"
            )

        if not encoder_num.include_bias:
            raise ValueError(
                "The numerical encoder should have `include_bias=True`. This is "
                "necessary so that so that it is possible to model a different "
                "intercept for each categorical feature's level."
            )

        self.encoder_cat = encoder_cat
        self.encoder_num = encoder_num

    def fit(self, X, y=None):
        """
        Args:
            X : pd.DataFrame of shape (n_samples, n_features)
                The data to fit.
            y : None
                Ignored.

        Returns:
            self : object
                Fitted transformer.
        """
        try:
            check_is_fitted(self.encoder_cat, "fitted_")
        except NotFittedError:
            self.encoder_cat.fit(X, y)

        try:
            check_is_fitted(self.encoder_num, "fitted_")
        except NotFittedError:
            self.encoder_num.fit(X)

        self.n_features_out_ = (
            self.encoder_cat.n_features_out_ * self.encoder_num.n_features_out_
        )
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.
        """
        check_is_fitted(self, "fitted_")
        X_cat = self.encoder_cat.transform(X)
        X_num = self.encoder_num.transform(X)
        # guard for single category
        if X_cat.shape[1] == 1:
            return X_num
        else:
            return tensor_product(X_cat, X_num)


class ICatSplineEncoder(TransformerMixin, BaseEstimator):
    """Encode the interaction between one categorical and one spline-encoded numerical
    feature. This encoder can also work with a cyclical numerical feature.

    Args:
        encoder_cat : eensight.features.encode.CategoricalEncoder
            The encoder for the categorical feature. It must encode features in
            an one-hot form.
        encoder_num : eensight.features.encode.SplineEncoder or
            eensight.features.encode.CyclicalFeatures
            The encoder for the numerical feature.

    Notes:
        - If the categorical encoder is already fitted, it will not be re-fitted during
        `fit` or `fit_transform`. 
        - The numerical encoder will always be (re)fitted (one encoder per level of 
        categorical feature).
        - If the numerical encoder is a `SplineEncoder`, it should have `include_bias=True`. 
        This is necessary so that so that it is possible to model a different intercept for 
        each categorical feature's level.
    """

    def __init__(
        self,
        *,
        encoder_cat: CategoricalEncoder,
        encoder_num: Union[SplineEncoder, CyclicalFeatures],
    ):
        if encoder_cat.encode_as != "onehot":
            raise ValueError(
                "This encoder supports only one-hot encoding of the "
                "categorical feature"
            )

        if isinstance(encoder_num, SplineEncoder) and not encoder_num.include_bias:
            raise ValueError(
                "The numerical encoder should have `include_bias=True`. This is "
                "necessary so that so that it is possible to model a different "
                "intercept for each categorical feature's level."
            )

        self.encoder_cat = encoder_cat
        self.encoder_num = encoder_num

    def fit(self, X, y=None):
        """
        Args:
            X : pd.DataFrame of shape (n_samples, n_features)
                The data to fit.
            y : None
                Ignored.

        Returns:
            self : object
                Fitted transformer.
        """
        try:
            check_is_fitted(self.encoder_cat, "fitted_")
        except NotFittedError:
            self.encoder_cat.fit(X, y)

        encoders = OrderedDict({})
        cat_features = pd.DataFrame(data=self.encoder_cat.transform(X), index=X.index)

        for i, col in enumerate(cat_features.columns):
            mask = cat_features[col] == 1
            enc = clone(self.encoder_num)
            try:
                encoders[i] = enc.fit(X.loc[mask])
            except ValueError:
                encoders[i] = enc

        self.num_encoders_ = encoders
        self.n_features_out_ = (
            len(encoders) * next(iter(encoders.values())).n_features_out_
        )
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.

        Returns:
            X_transformed : numpy array, shape (n_samples, n_features_out_)
                The selected column subset as a numpy array.
        """
        check_is_fitted(self, "fitted_")

        out = None
        cat_features = pd.DataFrame(data=self.encoder_cat.transform(X), index=X.index)

        for i, encoder in self.num_encoders_.items():
            mask = cat_features.loc[:, i] == 1
            subset = X.loc[mask]
            if subset.empty or not encoder.fitted_:
                trf = pd.DataFrame(
                    data=np.zeros((X.shape[0], encoder.n_features_out_)), index=X.index
                )
            else:
                trf = pd.DataFrame(
                    data=encoder.transform(subset), index=subset.index
                ).reindex(X.index, fill_value=0)
            out = pd.concat((out, trf), axis=1)

        out = np.array(out)
        return out
