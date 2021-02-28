# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os 
import math 
import pickle
import optuna
import warnings

import pandas as pd 
import numpy as np 

from datetime import datetime
from types import SimpleNamespace
from collections.abc import Iterable
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error as msqe
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from catboost import Pool, CatBoostClassifier, CatBoostRegressor, cv
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter



warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(0)


BINCLASS_LOSSES = ['Logloss', 'CrossEntropy']
MULTICLASS_LOSSES = ['MultiClass', 'MultiClassOneVsAll']
CLASSIFICATION_LOSSES = BINCLASS_LOSSES + MULTICLASS_LOSSES
REGRESSION_LOSSES = ['MAE', 'MAPE', 'Quantile', 'RMSE']

DEFAULT_PARAMS = {
        'iterations': 2000,
        'learning_rate': 0.1,
        'depth': 4,
        'l2_leaf_reg': 10,
        'bootstrap_type': 'Bayesian',
        'od_wait': 100,
        'od_type': 'Iter',
        'task_type': 'CPU',
        'has_time': False 
}


#################### Auxiliary functions #############################################
######################################################################################

def filter_cv_folds(X, y, folds, exclude_train=None, exclude_test=None):
    n_obs = X.shape[0]
    positions = pd.Series(data=range(n_obs), index=X.index)

    for train_index, eval_index in folds.split(X, y):
        if exclude_train is not None:
            exclude_train_idx = positions[X.index.isin(exclude_train)].values
            train_index = np.setdiff1d(train_index, exclude_train_idx)
        
        if exclude_test is not None:
            exclude_test_idx = positions[X.index.isin(exclude_test)].values
            eval_index = np.setdiff1d(eval_index, exclude_test_idx)        

        yield train_index, eval_index


def fit_and_score(estimator, X_train, y_train, X_eval, y_eval, scorers, cat_features):
    """Fit estimator and compute scores for a given dataset split."""    
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    eval_pool = Pool(data=X_eval, label=y_eval, cat_features=cat_features)
    score_pool = Pool(data=X_eval, cat_features=cat_features)
    
    estimator.fit(train_pool, eval_set=eval_pool, use_best_model=True, verbose=False)
    y_pred = estimator.predict(score_pool)

    result = {
        name: scorer(y_eval, y_pred) for name, scorer in scorers.items()
    }
    result.update({'iterations': estimator.tree_count_})
    return result


#################### BaseBoostedTree #################################################
######################################################################################


class BaseBoostedTree(BaseEstimator):
    def __init__(self, **params):
        self.cat_features = params.pop('cat_features', [])
        # Duplicates are resolved in favor of the value in params
        self._estimator_params = dict(DEFAULT_PARAMS, **params)


    def get_params(self, deep=True):
        return dict(self._estimator_params)


    def set_params(self, **params):
        if not params:
            return self
        self._estimator_params.update(**params)
        return self

    
    def optimize(self, X, y, optimizing_direction='min',
                             n_trials=10,
                             n_splits=5, 
                             verbose=False):
        
        def _objective(trial):
            param_space = {
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 2, 30),
                'learning_rate': trial.suggest_float('learning_rate', 1e-2, 5e-1, log=True),
                'depth': trial.suggest_int('depth', 1, 8)
            }

            fit_params = self.get_params()
            fit_params.update(param_space)
            # To speed up
            fit_params.update({'bootstrap_type': 'Bernoulli', 'subsample': 0.3})
            
            cv_data = self.cross_validate(X, y, n_splits=n_splits, 
                                                fit_params=fit_params, 
                                                verbose=verbose)
            loss_function = self.get_params()['loss_function']
            
            if optimizing_direction == 'min':
                best_value = np.min(cv_data[f'test-{loss_function}-mean'])
                best_iter = np.argmin(cv_data[f'test-{loss_function}-mean'])
            else:
                best_value = np.max(cv_data[f'test-{loss_function}-mean'])
                best_iter = np.argmax(cv_data[f'test-{loss_function}-mean'])
            
            trial.set_user_attr('best_iter', best_iter)
            return best_value

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True), 
                                    pruner=optuna.pruners.SuccessiveHalvingPruner())
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=verbose)
        
        self.set_params(**study.best_params)
        return SimpleNamespace(best_params=study.best_params, 
                               best_iter=study.best_trial.user_attrs['best_iter']
        )
    
    
    def cross_validate(self, X, y, fit_params=None, 
                                   n_splits=5, 
                                   custom_metric=None, 
                                   metric_period=None, 
                                   exclude_train=None, 
                                   exclude_test=None,
                                   plot=False, 
                                   verbose=False):
        if fit_params is None:
            fit_params = self.get_params()
        else:
            fit_params = dict(self.get_params(), **fit_params)
        
        if fit_params['loss_function'] in CLASSIFICATION_LOSSES:
            folds = KFold(n_splits=n_splits, shuffle=True)
        else:
            time_step = X.index.to_series().diff().min()
            steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)
            folds = TimeSeriesSplit(n_splits=n_splits, gap=24*steps_per_hour)
        
        if (exclude_train is not None) or (exclude_test is not None):
            folds = filter_cv_folds(X, y, folds, exclude_train=exclude_train, 
                                                 exclude_test=exclude_test)

        if custom_metric is not None:
            fit_params.update(dict(custom_metric=custom_metric))
        
        if fit_params['loss_function'] in CLASSIFICATION_LOSSES:
            classes = np.unique(y)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            fit_params.update(dict(class_weights=dict(zip(classes, weights))))
        
        cv_data = cv(pool = Pool(X, label=y, cat_features=self.cat_features),
                     params = fit_params,
                     folds=folds,
                     metric_period=metric_period,
                     plot=plot,
                     verbose=verbose
        )
        return cv_data
    
    
    def fit(self, X, y, iterations=None, plot=False, verbose=False):
        fit_params = self.get_params()
        
        if fit_params['loss_function'] in CLASSIFICATION_LOSSES:
            classes = np.unique(y)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=np.array(y))
            fit_params.update(dict(class_weights=dict(zip(classes, weights))))
        
        if iterations is not None:
            fit_params.update(dict(iterations=iterations))
        
        self.base_estimator_ = self._make_estimator(**fit_params)
        self.base_estimator_.fit(Pool(data=X, label=y, cat_features=self.cat_features), 
                                        plot=plot, verbose=verbose)

        self.target_name_ = None
        if isinstance(y, pd.DataFrame):
            self.target_name_ = y.columns.tolist()
        elif isinstance(y, pd.Series):
            self.target_name_ = [y.name]
        
        self.is_fitted_ = True
        return self


    def predict(self, X):
        check_is_fitted(self, 'is_fitted_') 
        pred = self.base_estimator_.predict(Pool(data=X, cat_features=self.cat_features))
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.DataFrame(data=pred, columns=self.target_name_, index=X.index)
        else:
            return pred
    
    
    def fit_predict(self, X, y, iterations=None, plot=False, verbose=False):
        self.fit(X, y, iterations=iterations, plot=plot, verbose=verbose)
        return self.predict(X)

    def save_model(self, name, model_dir, metadata=None):
        check_is_fitted(self, 'is_fitted_') 
        
        name = '_'.join([name, datetime.now().strftime("%Y_%m_%d_%H_%M_%S")])
        self.base_estimator_.save_model(os.path.join(model_dir, f'{name}.cbm'))
        
        if metadata is not None:
            with open(os.path.join(model_dir, f'{name}_meta.pickle'), 'wb') as handle:
                pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self 


###########################################################################################
################# BoostedTreeClassifier ###################################################
###########################################################################################


class BoostedTreeClassifier(ClassifierMixin, BaseBoostedTree):
    def __init__(self, **params):
        task = params.pop('task', 'classification')
        if 'multi' in task:
            params.update(dict(loss_function='MultiClassOneVsAll'))
        else:
            params.update(dict(loss_function='Logloss')) 
        super().__init__(**params)

    
    def _make_estimator(self, **params):
        if params:
            return CatBoostClassifier(**dict(self.get_params(), **params))
        else:
            return CatBoostClassifier(**self.get_params())
    

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_') 
        pred = self.base_estimator_.predict_proba(Pool(data=X, cat_features=self.cat_features))

        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.DataFrame(data=pred, columns=self.base_estimator_.classes_, index=X.index)
        else:
            return pred 


    def evaluate_conformity(self, X, y):
        prediction = self.predict_proba(X) 
        return prediction.max(axis=1) - prediction.apply(lambda x: x[y[x.name.date()]], axis=1)



###########################################################################################
################# BoostedTreeRegressor ###################################################
###########################################################################################

class BoostedTreeRegressor(RegressorMixin, BaseBoostedTree):
    def __init__(self, **params):
        task = params.pop('task', 'regression')
        if 'multi' in task:
            params.update(dict(loss_function='MultiRMSE')) 
        else:
            params.update(dict(loss_function='RMSE')) 
        params.update(dict(has_time=True))
        super().__init__(**params)


    def _make_estimator(self, **params):
        if params:
            return CatBoostRegressor(**dict(self.get_params(), **params))
        else:
            return CatBoostRegressor(**self.get_params())
    
    
    def score_over_folds(self, X, y, n_splits=5, 
                                     scorers=None, 
                                     fit_params=None, 
                                     exclude_train=None, 
                                     exclude_test=None, 
                                     calibrate=False):
        
        time_step = X.index.to_series().diff().min()
        steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)
        folds = TimeSeriesSplit(n_splits=n_splits, gap=24*steps_per_hour)
        
        has_metrics = True
        if (scorers is None) and not calibrate:
            scorers = {
                'CVRMSE': lambda y_true, y_pred: msqe(y_true, y_pred, squared=False)/np.mean(y_true),
                'NMBE': lambda y_true, y_pred: np.sum(y_true - y_pred) / (np.mean(y_true)*len(y_true))
            }
        elif (scorers is not None) and calibrate:
            scorers.update({'NC': lambda y_true, y_pred: y_true - y_pred})
        elif (scorers is None) and calibrate:
            has_metrics = False
            scorers = {
                'NC': lambda y_true, y_pred: y_true - y_pred
            }
        
        if fit_params is None:
            fit_params = self.get_params()
        else:
            fit_params = dict(self.get_params(), **fit_params)
        
        scores = []
        estimator = self._make_estimator(**fit_params)
        
        for train_index, eval_index in filter_cv_folds(X, y, folds, exclude_train=exclude_train, 
                                                                    exclude_test=exclude_test):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index] 
            X_eval, y_eval = X.iloc[eval_index], y.iloc[eval_index]
            
            scores.append(
                fit_and_score(estimator, X_train, y_train, X_eval, y_eval, scorers, self.cat_features)
            )
        
        cal_scores = []
        metrics = []

        if calibrate:
            for item in scores:
                cal_scores.append(item.pop('NC'))
            cal_scores = np.concatenate(cal_scores, axis=0)

        if has_metrics:
            metrics = pd.DataFrame(scores)

        return SimpleNamespace(metrics=metrics, cal_scores=cal_scores)


    def calibrate(self, cal_scores):
        min_val = 1.1 * cal_scores.min()
        max_val = 1.1 * cal_scores.max()
        ecdf = ECDF(cal_scores)
        self.empirical_cdf_ = monotone_fn_inverter(ecdf, np.linspace(min_val, max_val, 1000))
        return self 

    
    def _get_intervals(self, prediction, significance):
        n_test = len(prediction)
        intervals = np.zeros((n_test, 2))
        intervals[:, 0] = prediction.iloc[:,0] + self.empirical_cdf_(significance/2)
        intervals[:, 1] = prediction.iloc[:,0] + self.empirical_cdf_(1-significance/2)
        return intervals
    
    
    def predict_uncertainty(self, X, significance=None):
        """Constructs prediction intervals for a set of test examples."""
        if not hasattr(self, 'empirical_cdf_'):
            raise RuntimeError('The model has not been calibrated for conformal predictions.')

        if significance is None:
            significance = np.arange(0.01, 1.0, 0.01)
        elif not isinstance(significance, Iterable):
            significance = [significance]
        
        n_test = len(X)
        prediction = self.predict(X)
        intervals = np.zeros((n_test, 2, len(significance)))

        for i, s in enumerate(significance):
            predictions = self._get_intervals(prediction, s)
            intervals[:, :, i] = predictions

        if isinstance(X, (pd.DataFrame, pd.Series)):
            return {
                round(s, 2): pd.DataFrame(
                               data=intervals[:, :, i], 
                               columns=['lower', 'upper'], 
                               index=X.index) 
                    for i, s in enumerate(significance)
            }
        else:
            return {round(s, 2): intervals[:, :, i] for i, s in enumerate(significance)}

        
