# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from collections import OrderedDict

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
    maybe_reshape_2d,
    tensor_product,
)

logger = logging.getLogger("feature-encoding")

UNKNOWN_VALUE = -1

#####################################################################################
# Encode features
# All encoders generate numpy arrays
#####################################################################################

# ------------------------------------------------------------------------------------
# IdentityEncoder (utility encoder)
# ------------------------------------------------------------------------------------


class IdentityEncoder(TransformerMixin, BaseEstimator):
    """
    The identity encoder returns what it is fed.

    Args:
        feature : str or list of str, default=None
            The name(s) of the input dataframe's column(s) to return. If
            None, the whole input dataframe will be returned.
        as_filter : bool, default=False
            If True, the encoder will return all feature labels for which
            "feature in label == True".
        include_bias : bool, default=False
            If True, a column of ones is added to the output.
    """

    def __init__(self, feature=None, as_filter=False, include_bias=False):
        if as_filter and isinstance(feature, list):
            raise ValueError(
                "If `as_filter` is True, `feature` cannot include multiple feature names"
            )

        self.feature = feature
        self.as_filter = as_filter
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

        if self.feature is None:
            n_features_out_ = X.shape[1]
        elif (self.feature is not None) and not self.as_filter:
            n_features_out_ = len(self.features_)
        else:
            n_features_out_ = X.filter(like=self.feature, axis=1).shape[1]

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
        X = check_X(X)

        if (self.feature is not None) and not self.as_filter:
            X = X[self.features_]
        elif self.feature is not None:
            X = X.filter(like=self.feature, axis=1)

        if self.include_bias:
            X = add_constant(X, has_constant="raise")

        return np.array(X)


# ------------------------------------------------------------------------------------
# Encode categorical data
# ------------------------------------------------------------------------------------


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

    def _to_pandas(self, x):
        return pd.DataFrame(x, columns=[self.feature])

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
                        FunctionTransformer(self._to_pandas),
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


# ------------------------------------------------------------------------------------
# Encode pairwise categorical data interactions
# ------------------------------------------------------------------------------------


class ICatEncoder(TransformerMixin, BaseEstimator):
    """Encode the interaction between two categorical features

    Args:
        encoder_left : eensight.features.encode.CategoricalEncoder
            The encoder for the first of the two features
        encoder_right : eensight.features.encode.CategoricalEncoder
            The encoder for the second of the two features

    Note:
        - Both encoders should have the same `encode_as` parameter.
        - If one or both of the encoders is already fitted, it will not be
        re-fitted during `fit` or `fit_transform`.
    """

    def __init__(
        self, encoder_left: CategoricalEncoder, encoder_right: CategoricalEncoder
    ):
        if (not isinstance(encoder_left, CategoricalEncoder)) or (
            not isinstance(encoder_right, CategoricalEncoder)
        ):
            raise ValueError(
                "This pairwise interaction encoder expects `CategoricalEncoder` encoders"
            )
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


# ------------------------------------------------------------------------------------
# Encode numerical data
# ------------------------------------------------------------------------------------


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
        include_bias : bool, default=True
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
        include_bias=True,
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


# ------------------------------------------------------------------------------------
# Encode pairwise interactions between numerical features
# ------------------------------------------------------------------------------------


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
        if (not isinstance(encoder_left, SplineEncoder)) or (
            not isinstance(encoder_right, SplineEncoder)
        ):
            raise ValueError(
                "This pairwise interaction encoder expects `SplineEncoder` encoders"
            )

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
        if (not isinstance(encoder_left, IdentityEncoder)) or (
            not isinstance(encoder_right, IdentityEncoder)
        ):
            raise ValueError(
                "This pairwise interaction encoder expects `IdentityEncoder` encoders"
            )

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
            finally:
                if encoder.n_features_out_ > 1:
                    raise ValueError(
                        "This pairwise interaction encoder supports only single-feature encoders"
                    )

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


# ------------------------------------------------------------------------------------
# Encode pairwise interactions of one numerical and one categorical feature
# ------------------------------------------------------------------------------------


class ICatLinearEncoder(TransformerMixin, BaseEstimator):
    """Encode the interaction between one categorical and one linear numerical
    feature.

    Args:
        encoder_cat : eensight.features.encode.CategoricalEncoder
            The encoder for the categorical feature. It must encode features in
            an one-hot form.
        encoder_num : eensight.features.encode.IdentityEncoder
            The encoder for the numerical feature.

    Note:
        If either encoder is already fitted, it will not be re-fitted during `fit`
        or `fit_transform`.
    """

    def __init__(
        self, *, encoder_cat: CategoricalEncoder, encoder_num: IdentityEncoder
    ):
        if not isinstance(encoder_cat, CategoricalEncoder):
            raise ValueError("`encoder_cat` must be a CategoricalEncoder")

        if encoder_cat.encode_as != "onehot":
            raise ValueError(
                "This encoder supports only one-hot encoding of the "
                "categorical feature"
            )

        if not isinstance(encoder_num, IdentityEncoder):
            raise ValueError("`encoder_num` must be an IdentityEncoder")

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
    feature.

    Args:
        encoder_cat : eensight.features.encode.CategoricalEncoder
            The encoder for the categorical feature. It must encode features in
            an one-hot form.
        encoder_num : eensight.features.encode.SplineEncoder
            The encoder for the numerical feature.

    Notes:
        - If the categorical encoder is already fitted, it will not be re-fitted during
        `fit` or `fit_transform`.
        - The numerical encoder will always be (re)fitted (one encoder per level of
        categorical feature).
    """

    def __init__(
        self,
        *,
        encoder_cat: CategoricalEncoder,
        encoder_num: SplineEncoder,
    ):
        if not isinstance(encoder_cat, CategoricalEncoder):
            raise ValueError("`encoder_cat` must be a CategoricalEncoder")

        if encoder_cat.encode_as != "onehot":
            raise ValueError(
                "This encoder supports only one-hot encoding of the "
                "categorical feature"
            )

        if not isinstance(encoder_num, SplineEncoder):
            raise ValueError("`encoder_num` must be a SplineEncoder")

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
            if subset.empty or (not encoder.fitted_):
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
