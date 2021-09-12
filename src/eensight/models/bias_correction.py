# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d
from statsmodels.distributions.empirical_distribution import ECDF

from eensight.utils import check_X


class QuantileMatch(TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, 1)
                The model's predictions.
            y : pd.DataFrame, shape (n_samples, 1)
                The actual onservations

        Returns:
            self : object
                Returns self.
        """
        self.cdf_ = ECDF(column_or_1d(X))
        self.mod_data_ = column_or_1d(X)
        self.obs_data_ = column_or_1d(y)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, 1)
                The model's predictions to correct for bias

        Returns:
            transformed : pd.DataFrame
        """
        check_is_fitted(self, "fitted_")
        X = check_X(X)
        index = X.index
        columns = X.columns
        X = column_or_1d(X)

        p = self.cdf_(X) * 100
        cor = np.percentile(self.obs_data_, p) - np.percentile(self.mod_data_, p)
        return pd.DataFrame(data=(X + cor), index=index, columns=columns)
