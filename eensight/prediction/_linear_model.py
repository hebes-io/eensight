# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import optuna
import logging
import warnings
import traceback

import numpy as np
import pandas as pd 

from math import floor 
from joblib import Parallel
from sklearn.base import clone 
from types import SimpleNamespace
from sklearn.utils.fixes import delayed
from sklearn.base import RegressorMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.model_selection import KFold
from sklearn.model_selection import check_cv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from category_encoders.one_hot import OneHotEncoder
from sklearn.metrics import mean_squared_error as msqe
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from sklego.preprocessing import ColumnSelector, IntervalEncoder, PatsyTransformer

from eensight.preprocessing.utils import as_list
from eensight.preprocessing._day_typing import DateFeatureTransformer
from eensight.prediction.utils import validate_input_data, validate_target_data


warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(0)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def fit_and_score(estimator, 
                  X_train, 
                  y_train, 
                  X_test, 
                  y_test, 
                  scorers, 
                  fit_params=None, 
                  return_estimator=False):
    """Fit estimator and compute scores for a given dataset split.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X_train : array-like of shape (n_samples, n_features)
        The training data to fit.
    y_train : array-like of shape (n_samples,)
        The training target data.
    X_test : array-like of shape (n_samples, n_features)
        The evaluation data for out-of-sample prediction.
    y_test : array-like of shape (n_samples,)
        The testing target data.
    scorers : A dict mapping scorer name to a callable. The callable 
        object / fn should have signature ``scorer(y_true, y_pred)``.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    """    
    fit_params = fit_params if fit_params is not None else {}

    try:
        estimator.fit(X_train, y_train, **fit_params)
    except Exception as e:
        logger.warn(f'Estimator fit failed: {traceback.format_exc().splitlines()[-1]}. '
                    'The score on this train-test partition for these parameters will '
                    f'be set to {np.nan}. '
        )
        result = {
            name: np.nan for name in scorers
        }
        
    else:
        y_pred = estimator.predict(X_test)
        result = {
            name: np.array(scorer(y_test.loc[y_pred.index], y_pred)) for name, scorer in scorers.items()
        }
    
    if return_estimator:
        result['estimator'] = estimator

    return result



def cross_validate(estimator, X, y, groups=None, 
                                    scorers=None, 
                                    cv=None,
                                    n_jobs=None, 
                                    verbose=0,
                                    fit_params=None,
                                    pre_dispatch='2*n_jobs', 
                                    return_estimator=False):
    """Evaluate metric(s) by cross-validation.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,)
        The target data.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    scorers : A dict mapping scorer name to a callable. The callable 
        object / fn should have signature ``scorer(y_true, y_pred)``.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
    """
    
    cv = check_cv(cv, y)
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(delayed(fit_and_score)(clone(estimator), 
                                                X.iloc[train], y.iloc[train], 
                                                X.iloc[test], y.iloc[test], 
                                                scorers, 
                                                fit_params=fit_params, 
                                                return_estimator=return_estimator
                        ) for train, test in cv.split(X, y, groups))
    
    estimators = None
    if return_estimator:
        estimators = [item.pop('estimator') for item in results]
    
    scores = {
        key: np.asarray([item[key] for item in results]).flatten() for key in results[0]
    }
    return SimpleNamespace(scores=scores, estimators=estimators)


def towt_distance_metric(x, y):
    td = x - y
    return floor(np.abs(td.total_seconds()) / 86400)


#####################################################################################
################## TOWTPredictor ####################################################
#####################################################################################


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
        of 0.5 provides weights that include all months but only 8 of them have
        weights over 0.5, and a value of 1 keeps all months above a 0.65 weight.
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
    
    
    def _create_anchors(self, X):
        self.anchors_ = []

        start = X.index.min() 
        end = X.index.max()
        diff = (end - start) / (2*self.n_estimators)
        for i in range(1, 2*self.n_estimators, 2):
            self.anchors_.append(start + diff*i)
        #assert len(self.anchors_) == self.n_estimators
    
    
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

    
    def optimize(self, X, y, budget=10, 
                             n_splits=None,
                             shuffle=False, 
                             random_state=None, 
                             verbose=False):
         
        
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        scorer = {
            'CVRMSE': lambda y_true, y_pred: msqe(y_true, y_pred, squared=False)/np.mean(y_true),
        }

        def _objective(trial):
            param_space = {
                'n_estimators' : trial.suggest_int('n_estimators', 1, 12),
                'sigma' : trial.suggest_float('sigma', 0.1, 1.5),
               
            }
            
            estimator = clone(self)
            estimator.set_params(**param_space)
            cv_results = cross_validate(estimator, X, y, scorers=scorer, 
                                                         cv=cv, 
                                                         n_jobs=None, 
                                                         verbose=verbose
            )
            return np.mean(cv_results.scores['CVRMSE'])
            
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True),
                                    pruner=optuna.pruners.SuccessiveHalvingPruner())
        study.optimize(_objective, n_trials=budget, show_progress_bar=verbose)
        
        self.set_params(**study.best_params)
        self.optimized_ = True
        return SimpleNamespace(study=study, 
                               best_params=study.best_params, 
                               best_value=study.best_trial.value
        )

    
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
            self._create_anchors(X)
            for anchor in self.anchors_:
                distances = np.array([towt_distance_metric(anchor, idx) for idx in X.index])
                distances = MinMaxScaler().fit_transform(distances.reshape(-1,1)).squeeze()
                sample_weight = np.exp(-(distances**2) / (2*self.sigma**2))
                
                estimator = self._make_estimator()
                estimator.fit(features, y, sample_weight=sample_weight)
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
            self.anchors_
        
        except AttributeError:
            estimator = self.estimators_[0]
            prediction = estimator.predict(features)
            prediction = pd.DataFrame(data=prediction, index=X.index, columns=[self.target_name_])
            
        else:
            n_obs = len(X)
            weights = np.ones((n_obs, self.n_estimators))
            for i, anchor in enumerate(self.anchors_):
                distances = np.array([towt_distance_metric(anchor, idx) for idx in X.index])
                distances = MinMaxScaler().fit_transform(distances.reshape(-1,1)).squeeze()
                sample_weight = np.exp(-(distances**2) / (2*self.sigma**2))
                weights[:, i] = sample_weight

            total = weights.sum(axis=1).reshape(-1,1)
            weights = pd.DataFrame(data=weights / total, index=X.index)
            
            prediction = pd.Series(0, index=X.index)
            for i, estimator in enumerate(self.estimators_):
                pred = estimator.predict(features)
                pred = pd.Series(data=pred.squeeze(), index=X.index)
                pred = pred.multiply(weights.iloc[:, i])
                prediction = prediction + pred
            
            prediction = prediction.to_frame(self.target_name_)

        return prediction