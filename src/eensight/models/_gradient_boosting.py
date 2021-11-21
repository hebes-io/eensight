# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

DEFAULT_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 1000,
    "learning_rate": None,
    "depth": 4,
    "l2_leaf_reg": 10,
    "bootstrap_type": "Bayesian",
    "od_wait": 50,
    "od_type": "Iter",
    "task_type": "CPU",
    "has_time": True,
    "verbose": False,
    "allow_writing_files": False,
}


class BoostedTreeRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, **params):
        self.cat_features = params.pop("cat_features", [])
        # Duplicates are resolved in favor of the value in params
        self.estimator_params_ = dict(DEFAULT_PARAMS, **params)

    def set_params(self, **params):
        if not params:
            return self
        self.estimator_params_.update(**params)
        return self

    def fit(self, X, y, iterations=None, eval_set=None, verbose=False):
        fit_params = dict(self.estimator_params_)

        if iterations is not None:
            fit_params.update(dict(iterations=iterations))

        self.base_estimator_ = CatBoostRegressor(**fit_params)
        self.base_estimator_.fit(
            X,
            y=y,
            cat_features=self.cat_features,
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else None,
            verbose=verbose,
        )

        self.target_name_ = None
        if isinstance(y, pd.DataFrame):
            self.target_name_ = y.columns.tolist()
        elif isinstance(y, pd.Series):
            self.target_name_ = [y.name]

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        pred = self.base_estimator_.predict(
            Pool(data=X, cat_features=self.cat_features)
        )
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.DataFrame(data=pred, columns=self.target_name_, index=X.index)
        else:
            return pred

    def fit_predict(self, X, y, iterations=None, verbose=False):
        self.fit(X, y, iterations=iterations, verbose=verbose)
        return self.predict(X)
