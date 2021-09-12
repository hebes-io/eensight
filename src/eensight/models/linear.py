# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.validation import check_is_fitted

from eensight.features.compose import LinearModelFeatures
from eensight.utils import as_list, check_X, check_y


class LinearPredictor(RegressorMixin, BaseEstimator):
    """A linear regression model with flexible parameterization.

    Args:
        model_structure : dict
            The model configuration
        alpha : float (default=0.01)
            Regularization strength of the underlying ridge regression; must be a positive float.
            Regularization improves the conditioning of the problem and reduces the variance of
            the estimates. Larger values specify stronger regularization.
        fit_intercept : bool (default=False)
            Whether to fit the intercept for this model. If set to false, no intercept will be used
            in calculations.
    """

    def __init__(
        self, *, model_structure: Dict[str, Dict], alpha=0.01, fit_intercept=False
    ):
        self.model_structure = model_structure
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.composer_ = LinearModelFeatures(model_structure)

    @property
    def n_parameters(self):
        try:
            self.n_parameters_
        except AttributeError as exc:
            raise ValueError(
                "The number of parameters is acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            return self.n_parameters_

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        try:
            check_is_fitted(self, "base_estimator_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        X = check_X(X)
        y = check_y(y, index=X.index)
        self.target_name_ = y.columns[0]

        design_matrix = self.composer_.fit_transform(X, y)
        self.n_parameters_ = np.linalg.matrix_rank(design_matrix)

        if self.alpha is None:
            self.base_estimator_ = LinearRegression(fit_intercept=self.fit_intercept)
        else:
            self.base_estimator_ = Ridge(
                alpha=self.alpha, fit_intercept=self.fit_intercept
            )
        self.base_estimator_ = self.base_estimator_.fit(design_matrix, y)
        return self

    def predict(self, X: pd.DataFrame, include_components=False):
        check_is_fitted(self, "base_estimator_")
        X = check_X(X)
        design_matrix = self.composer_.transform(X)

        prediction = pd.DataFrame(
            data=self.base_estimator_.predict(design_matrix),
            columns=[self.target_name_],
            index=X.index,
        )

        if include_components:
            components = pd.DataFrame(
                0, index=X.index, columns=self.composer_.component_names_
            )
            feature_cols = self.composer_.component_matrix

            for col in components.columns:
                subset = feature_cols[feature_cols[col] == 1].index.to_list()
                coef = self.base_estimator_.coef_.squeeze()
                pred = np.matmul(design_matrix[:, subset], coef[subset])
                components[col] = components[col] + pred

            prediction = pd.concat((prediction, components), axis=1)

        return prediction
