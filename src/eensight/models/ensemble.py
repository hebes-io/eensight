# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class EnsemblePredictor(BaseEnsemble):
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
