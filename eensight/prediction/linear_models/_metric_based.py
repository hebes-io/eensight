# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
#import logging
import warnings

import numpy as np
import pandas as pd 

from datetime import datetime
from sklearn.base import RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaseEnsemble
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklego.preprocessing import ColumnSelector
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted
from sklearn.preprocessing import normalize, FunctionTransformer, KBinsDiscretizer
from sklego.preprocessing import ColumnSelector, IntervalEncoder, PatsyTransformer

from eensight.utils import as_list
from eensight.utils._prediction import validate_input_data, validate_target_data
from eensight.preprocessing._day_typing import DateFeatureTransformer, MMCFeatureTransformer

#logger = logging.getLogger(__file__)
#logger.setLevel(logging.INFO)

warnings.filterwarnings(action='ignore', category=FutureWarning)


def towt_identity(x):
    return pd.DataFrame(x.index, index=x.index)


def towt_metric(x: datetime, y: datetime):
    td = x - y
    return math.floor(np.abs(td.total_seconds()) / 86400) #difference in days



############################################################################################
################## AbstractDMPredictor #####################################################
############################################################################################


class AbstractDMPredictor(RegressorMixin, BaseEnsemble, abc.ABC):
    """Abstract class for TOWT-like linear regression models using a given distance metric 
    to predict energy consumption.

    Parameters
    ----------
    metric : callable
        The metric function used for computing the distance between two observations. It must 
        satisfy the following properties:
        Non-negativity: d(x, y) >= 0
        Identity: d(x, y) = 0 if and only if x == y
        Symmetry: d(x, y) = d(y, x)
        Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)
    transformer : any object that implements a `fit_transform` method
        The `fit_transform` method is used for transforming the datetime index of the input 
        `X` and of the anchors into a form that is understood by the distance metric.
    n_estimators : int (default=1)
        The number of estimators. This is equivalent to the number of segments in 
        the TOWT model. Each segment gives a higher statistical weight to the data
        subset that corresponds to it.
    sigma : float (default=0.5)
        It controls the kernel width that generates weights for the predictions. Generally, only 
        values between 0.1 and 2 make practical sense.
    temperature_col : str (default='temperature')
        The name of the column containing the temperature data. The dataframe passed to 
        `fit` and `predict` should have a column with the specified name.
    extra_regressors : str or list of str (default=None)
        The names of the additional regressors to be added to the model. The dataframe 
        passed to `fit` and `predict` should have a column with the specified names.
    missing : str (default='drop')
        Defines who missing values in input data are treated. It can be 'impute', 'drop' 
        or 'error'
    n_bins_temperature : int (default=5)
        The number of bins for approximating the temperature impact.
    """
    @_deprecate_positional_args
    def __init__(self, metric, 
                       transformer,
                       n_estimators=1,
                       sigma=0.5,
                       temperature_col='temperature',
                       extra_regressors=None,
                       missing='drop', 
                       n_bins_temperature=5
    ):
        super().__init__(
            base_estimator=LinearRegression(fit_intercept=False),
            n_estimators=n_estimators
        )

        self.metric = metric
        self.transformer = transformer
        self.sigma = sigma
        self.temperature_col = temperature_col
        self.extra_regressors = extra_regressors
        self.missing = missing
        self.n_bins_temperature = n_bins_temperature 
        self.extra_regressors_ = as_list(extra_regressors)
        

    def _generate_features(self, X, y=None, numeric_extra=None, categorical_extra=None):
        try:
            self.feature_pipeline_ 
        
        except AttributeError:
            n_days = X['dayofweek'].nunique()
            n_hours = X['hour'].nunique()
            
            self.feature_pipeline_ = Pipeline([
                ('features', FeatureUnion([
                        # time of week part of TOWT
                        ('weeks', Pipeline([
                                ('split', FeatureUnion([
                                        ('days', Pipeline([
                                                ('select', ColumnSelector('dayofweek')),
                                                ('ordinal', OrdinalEncoder(cols=['dayofweek'],
                                                                           return_df=False
                                                            )
                                                ),
                                                ('unknown', SimpleImputer(missing_values=-1, 
                                                                          strategy='most_frequent'
                                                            )
                                                )
                                            ])
                                        ),
                                        ('hours', Pipeline([
                                                ('select', ColumnSelector('hour')),
                                                ('ordinal', OrdinalEncoder(cols=['hour'],
                                                                           return_df=False
                                                            )
                                                ),
                                                ('unknown', SimpleImputer(missing_values=-1, 
                                                                          strategy='most_frequent'
                                                            )
                                                )
                                            ])
                                        )
                                    ])
                                ),
                                ('to_pandas', FunctionTransformer(lambda x: 
                                                pd.DataFrame(x, columns=['dayofweek', 'hour'])
                                              )
                                ),
                                ('term', PatsyTransformer('-1 + C(dayofweek):C(hour)'))
                            ])
                        ) if (n_days > 1) and (n_hours > 1) else 
                        
                        ('days', Pipeline([
                                    ('select', ColumnSelector('dayofweek')),
                                    ('ordinal', OrdinalEncoder(cols=['dayofweek'],
                                                               return_df=False
                                                )
                                    ),
                                    ('unknown', SimpleImputer(missing_values=-1, 
                                                              strategy='most_frequent'
                                                )
                                    ),
                                    ('to_pandas', FunctionTransformer(lambda x: 
                                                            pd.DataFrame(x, columns=['dayofweek'])
                                                 )
                                    ),
                                    ('one_hot', OneHotEncoder(cols=['dayofweek'], return_df=False)  
                                    )
                                ])
                        ) if n_days > 1 else 
                        
                        ('hours', Pipeline([
                                    ('select', ColumnSelector('hour')),
                                    ('ordinal', OrdinalEncoder(cols=['hour'],
                                                               return_df=False
                                                )
                                    ),
                                    ('unknown', SimpleImputer(missing_values=-1, 
                                                              strategy='most_frequent'
                                                )
                                    ),
                                    ('to_pandas', FunctionTransformer(lambda x: 
                                                            pd.DataFrame(x, columns=['hour'])
                                                 )
                                    ),
                                    ('one_hot', OneHotEncoder(cols=['hour'], return_df=False)  
                                    )
                                ])
                        ),

                        # temperature part of TOWT 
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

                        ('temperature_interact', 'drop' if n_hours == 1 else
                            Pipeline([
                                ('split', FeatureUnion([
                                        ('temperature_part', Pipeline([
                                                ('select', ColumnSelector(self.temperature_col)),
                                                ( 'create_bins', KBinsDiscretizer(
                                                                    n_bins=self.n_bins_temperature, 
                                                                    strategy='quantile', 
                                                                    encode='ordinal'
                                                                 ),
                                                )
                                            ])
                                        ),
                                        ('hour_part', Pipeline([
                                                ('select', ColumnSelector('hour')),
                                                ('ordinal', OrdinalEncoder(cols=['hour'],
                                                                           return_df=False
                                                           )
                                                ),
                                                ('unknown', SimpleImputer(missing_values=-1, 
                                                                          strategy='most_frequent'
                                                            )
                                                )
                                            ])
                                        )
                                    ])
                                ),
                                ('to_pandas', FunctionTransformer(lambda x: 
                                                pd.DataFrame(x, columns=[self.temperature_col, 'hour'])
                                            )
                                ),
                                ('term', PatsyTransformer(f'-1 + C({self.temperature_col}):C(hour)')
                                )
                            ])
                        ),
                        
                        # deal with extra numerical regressors
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

                        # deal with extra categorical regressors
                        ('categorical_regressors', 'drop' if not categorical_extra else 
                                TargetEncoder(cols=categorical_extra, 
                                              return_df=False, 
                                              handle_missing='value', 
                                              handle_unknown='value'
                                )              
                        ) 
                    ])
                )
            ])
            # Fit the pipeline
            self.feature_pipeline_.fit(X, y)

        finally:
            return self.feature_pipeline_.transform(X)

    
    @abc.abstractmethod
    def _get_anchors(self, X):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of AbstractDMPredictor"
            " and it must implement the `_get_anchors` method"
        )


    @abc.abstractmethod
    def metric_distance(self, u, v):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of AbstractDMPredictor"
            " and it must implement the `metric_distance` method"
        )

    
    def fit(self, X, y):
        try:
            check_is_fitted(self, 'fitted_')
        except NotFittedError:
            pass
        else:
            raise Exception('Estimator object can only be fit once. '
                            'Instantiate a new object.')

        X, categorical_extra, numeric_extra = validate_input_data(X, self.extra_regressors_, self.missing)
        y = validate_target_data(y, index=X.index)
        self.target_name_ = y.columns[0]

        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        features = self._generate_features(X_with_dates, y=y.iloc[:, 0], 
                                                         numeric_extra=numeric_extra, 
                                                         categorical_extra=categorical_extra
        )
        
        self._validate_estimator()   
        self.estimators_ = []
        self.anchors_ = self._get_anchors(X)
        self.cache_ = {}
        
        self.anchors_for_metric_ = self.transformer.fit_transform(self.anchors_)
        X_for_metric = self.transformer.fit_transform(X)
        
        for anchor, _ in self.anchors_for_metric_.iterrows():
            distances = np.ones(len(X))
            
            for i, idx in enumerate(X_for_metric.index):
                value = self.cache_.get((anchor.date(), idx.date()))
                if value is None:
                    value = self.metric_distance(self.anchors_for_metric_.loc[anchor], 
                                                 X_for_metric.loc[idx]
                    )
                    self.cache_[(anchor.date(), idx.date())] = value

                distances[i] = value

            sample_weight = np.exp(-(distances ** 2) / (2*(self.sigma * distances.std())**2))
            estimator = self._make_estimator()
            estimator.fit(features, y, sample_weight=sample_weight)
        
        self.fitted_ = True
        return self


    def predict(self, X):
        check_is_fitted(self, 'fitted_')

        X = validate_input_data(X, self.extra_regressors_, self.missing, fitting=False)
        X_with_dates = DateFeatureTransformer(remainder='passthrough').fit_transform(X)
        features = self._generate_features(X_with_dates)

        X_for_metric = self.transformer.fit_transform(X)
        weights = np.ones((len(X), self.n_estimators))
        
        for j, (anchor, _) in enumerate(self.anchors_for_metric_.iterrows()):
            for i, idx in enumerate(X_for_metric.index):
                value = self.cache_.get((anchor.date(), idx.date()))
                if value is None:
                    value = self.metric_distance(self.anchors_for_metric_.loc[anchor], 
                                                 X_for_metric.loc[idx]
                    )
                    self.cache_[(anchor.date(), idx.date())] = value

                weights[i, j] = value

        # Kernelize the weights
        weights = np.exp(-(weights ** 2) / (2*(self.sigma * weights.std())**2))
        # Normalize the weights
        weights = pd.DataFrame(data=normalize(weights, norm='l1', axis=1), index=X.index)

        prediction = pd.Series(0, index=X.index)
        for i, estimator in enumerate(self.estimators_):
            pred = estimator.predict(features)
            pred = pd.Series(data=pred.squeeze(), index=X.index)
            pred = pred.multiply(weights.iloc[:, i])
            prediction = prediction + pred

        prediction = prediction.to_frame(self.target_name_)
        return prediction

    
############################################################################################
################## TOWTPredictor ###########################################################
############################################################################################


class TOWTPredictor(AbstractDMPredictor):
    """A TOWT-like linear regression model

    Parameters
    ----------
    n_estimators : int (default=1)
        The number of estimators. This is equivalent to the number of segments in 
        the TOWT model. Each segment gives a higher statistical weight to the data
        subset that corresponds to it.
    sigma : float (default=0.5)
        It controls the kernel width that generates weights for the predictions. Generally, only 
        values between 0.1 and 3 make practical sense.
    temperature_col : str (default='temperature')
        The name of the column containing the temperature data. The dataframe passed to 
        `fit` and `predict` should have a column with the specified name.
    extra_regressors : str or list of str (default=None)
        The names of the additional regressors to be added to the model. The dataframe 
        passed to `fit` and `predict` should have a column with the specified names.
    missing : str (default='drop')
        Defines who missing values in input data are treated. It can be 'impute', 'drop' 
        or 'error'
    n_bins_temperature : int (default=5)
        The number of bins for approximating the temperature impact.
    """
    @_deprecate_positional_args
    def __init__(self, n_estimators=1,
                       sigma=0.5,
                       temperature_col='temperature',
                       extra_regressors=None,
                       missing='drop', 
                       n_bins_temperature=5
    ):
        
        super().__init__(
            metric=towt_metric,
            transformer=FunctionTransformer(func=towt_identity),
            n_estimators=n_estimators,
            sigma=sigma,
            temperature_col=temperature_col,
            extra_regressors=extra_regressors,
            missing=missing,
            n_bins_temperature=n_bins_temperature
        )


    def _get_anchors(self, X):
        anchors = []
    
        X = X.sort_index()
        start = X.index.min()
        end = X.index.max()
        diff = (end - start) / (2*self.n_estimators)
        
        for i in range(1, 2*self.n_estimators, 2):
            anchors.append(start + diff*i)
        assert len(anchors) == self.n_estimators
        return pd.Series(index=anchors, dtype='float')


    def metric_distance(self, u, v):
        return self.metric(u.item(), v.item())


############################################################################################
################## GTOWTPredictor ###########################################################
############################################################################################


class GTOWTPredictor(AbstractDMPredictor):
    """A generalized TOWT-like linear regression model using a custom distance metric 
    to predict energy consumption.

    Parameters
    ----------
    metric : callable
        The metric function used for computing the distance between two observations. It must 
        satisfy the following properties:
        Non-negativity: d(x, y) >= 0
        Identity: d(x, y) = 0 if and only if x == y
        Symmetry: d(x, y) = d(y, x)
        Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)
    anchors : list of datetime objects 
        One estimator per anchor will be fit, and each estimator will give a higher statistical 
        weight to the data subset that is close to the corresponding anchor.
    n_estimators : int (default=1)
        The number of estimators. This is equivalent to the number of segments in 
        the TOWT model. Each segment gives a higher statistical weight to the data
        subset that corresponds to it.
    sigma : float (default=0.5)
        It controls the kernel width that generates weights for the predictions. Generally, only 
        values between 0.1 and 3 make practical sense.
    temperature_col : str (default='temperature')
        The name of the column containing the temperature data. The dataframe passed to 
        `fit` and `predict` should have a column with the specified name.
    extra_regressors : str or list of str (default=None)
        The names of the additional regressors to be added to the model. The dataframe 
        passed to `fit` and `predict` should have a column with the specified names.
    missing : str (default='drop')
        Defines who missing values in input data are treated. It can be 'impute', 'drop' 
        or 'error'
    n_bins_temperature : int (default=5)
        The number of bins for approximating the temperature impact.
    """
    @_deprecate_positional_args
    def __init__(self, metric,
                       anchors,
                       n_estimators=1,
                       sigma=0.5,
                       temperature_col='temperature',
                       extra_regressors=None,
                       missing='drop', 
                       n_bins_temperature=5
    ):
        super().__init__(
            metric=metric,
            transformer=Pipeline([
                ('dates', DateFeatureTransformer()),
                ('features', MMCFeatureTransformer()),
            ]),
            n_estimators=n_estimators,
            sigma=sigma,
            temperature_col=temperature_col,
            extra_regressors=extra_regressors,
            missing=missing,
            n_bins_temperature=n_bins_temperature
        )
        self.anchors = anchors


    def _get_anchors(self, X):
        if isinstance(self.anchors, pd.DataFrame):
            return self.anchors[:self.n_estimators, 0].to_frame()
        elif isinstance(self.anchors, pd.Series):
            return self.anchors[:self.n_estimators].to_frame()

    
    def metric_distance(self, u, v):
        return self.metric(u, v)

