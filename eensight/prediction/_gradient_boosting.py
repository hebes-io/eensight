# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd 

from catboost import Pool, CatBoostRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args

from eensight.base import _BaseHeterogeneousEnsemble


DEFAULT_CB_PARAMS = {
        'iterations': 2000,
        'learning_rate': 0.1,
        'depth': 4,
        'l2_leaf_reg': 10,
        'bootstrap_type': 'Bayesian',
        'od_wait': 100,
        'od_type': 'Iter',
        'task_type': 'CPU',
        'has_time': True 
}



class GBRegressor(_BaseHeterogeneousEnsemble):
    @_deprecate_positional_args
    def __init__(self, **params):
        self.cat_features = params.pop('cat_features', [])
        # Duplicates are resolved in favor of the value in params
        self._estimator_params = dict(DEFAULT_CB_PARAMS, **params)
        self.estimators = [('estimator', CatBoostRegressor(**self._estimator_params))]

    
    def fit(self, X, y, sample_weight=None, iterations=None):
        if iterations is not None:
            self.set_params(**{'estimator__iterations': iterations})
        
        estimator = self.named_estimators['estimator']
        estimator.fit(Pool(data=X, 
                           label=y, 
                           cat_features=self.cat_features,
                           weight=sample_weight), 
                        verbose=False
        )

        self.target_name_ = None
        if isinstance(y, pd.DataFrame):
            self.target_name_ = y.columns.tolist()[0]
        elif isinstance(y, pd.Series):
            self.target_name_ = y.name
        
        self.fitted_ = True
        return self


    def predict(self, X):
        check_is_fitted(self, 'fitted_') 
        estimator = self.named_estimators['estimator']
        pred = estimator.predict(Pool(data=X, cat_features=self.cat_features))
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.Series(data=pred, name=self.target_name_, index=X.index)
        else:
            return pred
    
    
    def fit_predict(self, X, y, iterations=None):
        self.fit(X, y, iterations=iterations)
        return self.predict(X)

    






    
    
    
