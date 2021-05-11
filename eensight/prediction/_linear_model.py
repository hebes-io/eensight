# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import numpy as np
import pandas as pd 

from math import floor 
from sklearn.base import RegressorMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from category_encoders.one_hot import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from sklego.preprocessing import ColumnSelector, IntervalEncoder, PatsyTransformer

from eensight.preprocessing.utils import as_list
from eensight.preprocessing._day_typing import DateFeatureTransformer
from eensight.prediction.utils import validate_input_data, validate_target_data



warnings.simplefilter(action='ignore', category=FutureWarning)


class TOWTPredictor(RegressorMixin, BaseEnsemble):
    """TOWT-like linear regression model for predicting energy consumption.

    Parameters
    ----------
    base_estimator : A regressor (default=sklearn.linear_model.LinearRegression)
    fit_intercept : bool (default=True)
        Whether to calculate the intercept for this model. If set to False, no 
        intercept will be used in calculations.
    n_estimators : int (default=1)
        The number of estimators. This is equivalent to the number of segments in 
        the TOWT model. Each segment gives a higher statistical weight to the data
        subset that corresponds to it.
    estimator_params : tuple (default=('fit_intercept',))
        The name of the init parameters that are actually parameters for the regressor
    sigma : float (default=0.5)
        It controls the kernel width that generates weights for the predictions.
        A value of 0.1 provides non-zero values for a range of 3 months, a value
        of 0.5 provides weights that include all months but only a range of 6 have
        weights over 0.5, and a value of 1 keeps all months above a 0.5 weight.
    temperature_col : str (default='temperature')
        The name of the column containing the temperature data. The dataframe passed to `fit` and 
        `predict` should have a column with the specified name.
    extra_regressors : str or list of str (default=None)
        The names of the additional regressors to be added to the model. The dataframe passed to `fit` and 
        `predict` should have a column with the specified names.
    missing : str (default='drop')
        Defines who missing values in input data are treated. It can be 'impute', 'drop' or 'error'
    n_basis_teperature : int (default=7)
        The number of basis functons for approximating the temperature impact
    """
    @_deprecate_positional_args
    def __init__(self, base_estimator=LinearRegression(), 
                       fit_intercept=False,
                       n_estimators=1,
                       estimator_params=('fit_intercept',),
                       sigma=0.5,
                       temperature_col='temperature',
                       extra_regressors=None,
                       missing='drop', 
                       n_basis_teperature=7):
        self.base_estimator = base_estimator
        self.fit_intercept = fit_intercept
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        self.sigma = sigma
        self.temperature_col = temperature_col
        self.extra_regressors = extra_regressors
        self.missing = missing
        self.n_basis_teperature = n_basis_teperature 
        self.extra_regressors_ = as_list(extra_regressors)
    
    
    @staticmethod
    def time_distance(x, y):
        td = x - y
        return floor(np.abs(td.total_seconds()) / 86400) 
    
    
    def _generate_features(self, X, y=None, numeric_extra=None, categorical_extra=None):
        try:
            self.feature_pipeline_ 
        
        except AttributeError:
            n_days = X['dayofweek'].nunique()
            n_hours = X['hour'].nunique()
            
            self.feature_pipeline_ = Pipeline([
                ('features', FeatureUnion([
                        ('weeks', Pipeline([
                                    ('grab_relevant', ColumnSelector(['dayofweek', 'hour'])),
                                    ('term', PatsyTransformer(
                                                f'te(cr(dayofweek, df={n_days}), '
                                                   f'cr(hour, df={n_hours})) - 1'
                                            )
                                    )
                            ])
                        ) if (n_days > 1) and (n_hours > 1) else 
                        
                        ('days', OneHotEncoder(cols=['dayofweek'], 
                                                return_df=False, 
                                                handle_missing='value', 
                                                handle_unknown='value'
                                ) 
                        ) if n_days > 1 else 
                        
                        ('hours', OneHotEncoder(cols=['hour'], 
                                                return_df=False, 
                                                handle_missing='value', 
                                                handle_unknown='value'
                                )
                        ),

                        ('temperature', 
                            ColumnTransformer([
                                ( 'encode_temperature', 
                                  IntervalEncoder(n_chunks=10, 
                                                  span=0.1*X[self.temperature_col].std(), 
                                                  method='normal'
                                  ), 
                                  [self.temperature_col]
                                )
                                                
                            ])
                        ),

                        ('interactions_temperature', 'drop' if n_hours == 1 else
                            Pipeline([
                                ('split', FeatureUnion([
                                        ('temperature_part', ColumnTransformer([
                                                    ( 'create_bins', 
                                                      KBinsDiscretizer(n_bins=15, 
                                                                       strategy='kmeans', 
                                                                       encode='ordinal'
                                                      ), 
                                                      [self.temperature_col]
                                                    )
                                                    
                                                ])
                                        ),
                                        ('hour_part', Pipeline([
                                                        ('grab_hours', ColumnSelector(['hour']))
                                                ])
                                        )
                                    ])
                                ),
                                ('pandarize', FunctionTransformer(lambda x: 
                                                pd.DataFrame(x, columns=[self.temperature_col, 'hour'])
                                            )
                                ),
                                ('term', PatsyTransformer(
                                                f'te(cr({self.temperature_col}, df={self.n_basis_teperature}), '
                                                   f'cr(hour, df={n_hours})) - 1'
                                        )
                                )
                            ])
                        ),
                        
                        ('numerical_regressors', 'drop' if not numeric_extra else 
                                ColumnTransformer([
                                ( f'encode_{col}', 
                                  IntervalEncoder(n_chunks=4, 
                                                  span=0.1*X[col].std(), 
                                                  method='normal'
                                  ), 
                                  [col]
                                ) for col in numeric_extra                 
                            ])
                        ),

                        ('categorical_regressors', 'drop' if not categorical_extra else 
                                TargetEncoder(cols=categorical_extra, 
                                              return_df=False, handle_missing='value', handle_unknown='value'

                                )              
                        ) 
                    ])
                )
            ])
            
            self.feature_pipeline_.fit(X, y)

        finally:
            return self.feature_pipeline_.transform(X)


    def fit(self, X, y):
        X, categorical_extra, numeric_extra = validate_input_data(X, self.extra_regressors_, self.missing)
        
        if self.missing == 'drop':
            y = validate_target_data(y, index=X.index)
        else:
            y = validate_target_data(y)
        
        if not X.index.equals(y.index):
            raise ValueError('Inputs and target have different indices')
  
        self.target_name_ = y.columns[0]

        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        features = self._generate_features(X_with_dates, y=y.iloc[:, 0], 
                                                         numeric_extra=numeric_extra, 
                                                         categorical_extra=categorical_extra
        )
        
        self._validate_estimator()   
        self.estimators_ = []
        
        if (self.n_estimators is not None) and (self.n_estimators > 1):
            centers = []

            start = X.index.min() 
            end = X.index.max()
            diff = (end - start) / (2*self.n_estimators)
            for i in range(1, 2*self.n_estimators, 2):
                centers.append(start + diff*i)

            n_obs = len(X)
            self.weights_ = np.ones((n_obs, self.n_estimators))
            for i, center in enumerate(centers):
                distances = np.array([self.time_distance(center, idx) for idx in X.index])
                distances = MinMaxScaler().fit_transform(distances.reshape(-1,1)).squeeze()
                weights = np.exp(-(distances**2) / (self.sigma**2))
                estimator = self._make_estimator()
                estimator.fit(features, y, sample_weight=weights)
                self.weights_[:, i] = weights

            total = self.weights_.sum(axis=1).reshape(-1,1)
            self.weights_ = pd.DataFrame(data=self.weights_ / total, index=X.index)
            
        else:
            estimator = self._make_estimator()
            estimator.fit(features, y)

        self.fitted_ = True
        return self

    
    def predict(self, X):
        check_is_fitted(self, 'fitted_')
        X = validate_input_data(X, self.extra_regressors_, self.missing, fitting=False)
        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        features = self._generate_features(X_with_dates)
        
        try:
            self.weights_
        
        except AttributeError:
            estimator = self.estimators_[0]
            prediction = estimator.predict(features)
            prediction = pd.DataFrame(data=prediction, index=X.index, columns=[self.target_name_])
            
        else:
            prediction = pd.Series(0, index=X.index)
            for i, estimator in enumerate(self.estimators_):
                pred = estimator.predict(features)
                pred = pd.Series(data=pred.squeeze(), index=X.index)
                pred = pred.multiply(self.weights_.iloc[:, i])
                prediction = prediction + pred
            
            prediction = prediction.dropna().to_frame(self.target_name_)

        return prediction