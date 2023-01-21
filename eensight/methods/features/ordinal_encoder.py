# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Literal, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from eensight.utils import as_list, get_categorical_cols

UNKNOWN_VALUE = -1


class SafeOrdinalEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical features as an integer array.

    The encoder converts the categorical features into ordinal integers. This results
    in a single column of integers (0 to n_categories - 1) per feature.

    Args:
        features (str or list of str, optional): The names of the columns to encode. If
            None, all categorical columns will be encoded. Defaults to None.
        int_is_categorical (bool, optional): If True, integer types are considered categorical.
            Defaults to False.
        unknown_value (int, optional): This parameter will set the encoded value for unknown
            categories. It has to be distinct from the values used to encode any of the categories
            in `fit`. If None, the value `-1` is used. During `transform`, unknown categories will
            be replaced using the most frequent value along each column. Defaults to None.
        remainder ({'drop', 'passthrough'}, optional): By specifying ``remainder='passthrough'``,
            all the remaining columns of the input dataset will be automatically passed through
            (concatenated with the output of the transformer), otherwise, they will be dropped.
            Defaults to "passthrough".
    """

    def __init__(
        self,
        *,
        features: Union[str, List[str]] = None,
        int_is_categorical: bool = False,
        unknown_value: int = None,
        remainder: Literal["drop", "passthrough"] = "passthrough",
    ):
        if remainder not in ("passthrough", "drop"):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')

        self.features = features
        self.int_is_categorical = int_is_categorical
        self.unknown_value = unknown_value
        self.remainder = remainder

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder on the available data.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The features dataframe.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            SafeOrdinalEncoder: Fitted encoder.
        """
        if self.features is not None:
            cat_features = as_list(self.features)
            for name in cat_features:
                if pd.api.types.is_float_dtype(X[name]):
                    raise ValueError(
                        f"The encoder is applied on numerical data `{name}`"
                    )
        else:
            cat_features = get_categorical_cols(
                X, int_is_categorical=self.int_is_categorical
            )

        if X[cat_features].isnull().values.any():
            raise ValueError("Found NaN values in input's categorical data")

        self.feature_pipeline_ = Pipeline(
            [
                (
                    "select",
                    ColumnTransformer(
                        [("select", "passthrough", cat_features)], remainder="drop"
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
        self.cat_features_ = cat_features
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the encoder.

        Args:
            X (pandas.DataFrame of shape (n_samples, n_features)): The features
                dataframe.

        Raises:
            ValueError: If the input data does not pass the checks of `utils.check_X`.

        Returns:
            pandas.DataFrame: The encoded features dataframe.
        """
        check_is_fitted(self, "fitted_")
        out = pd.DataFrame(
            data=self.feature_pipeline_.transform(X),
            columns=self.cat_features_,
            index=X.index,
        )

        if self.remainder == "passthrough":
            columns = X.columns
            X = X.drop(self.cat_features_, axis=1)
            out = pd.concat((X, out), axis=1).reindex(columns=columns)

        return out
