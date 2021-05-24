# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import optuna
import warnings
import functools

import numpy as np 

from sklearn.base import clone 
from types import SimpleNamespace
from sklearn.model_selection import KFold

from eensight.prediction.model_selection import cross_validate


warnings.filterwarnings(action='ignore', category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(30)


def towt_objective(trial, *, estimator, 
                             data, 
                             n_estimators_max, 
                             sigma_max, 
                             cv, 
                             n_jobs, 
                             objective, 
                             verbose
):
    X, y = data

    param_space = {
        'n_estimators'        : trial.suggest_int('n_estimators', 1, n_estimators_max),
        'n_bins_temperature'  : trial.suggest_int('n_bins_temperature', 2, 7),
        'sigma'               : trial.suggest_float('sigma', 0.1, sigma_max),
        
    }
    
    estimator = clone(estimator)
    estimator.set_params(**param_space)
    cv_results = cross_validate(estimator, X, y, cv=cv, n_jobs=n_jobs, verbose=verbose)
    return np.mean(cv_results.scores[objective])



def towt_optimize(estimator, X, y, n_estimators_max=None, sigma_max=None, 
                    n_jobs=None, objective='CVRMSE', budget=20, n_splits=None, 
                    shuffle=False, random_state=None, verbose=False
):
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    _objective = functools.partial(towt_objective, 
                            estimator=estimator, 
                            data=(X,y), 
                            n_estimators_max=n_estimators_max, 
                            sigma_max=sigma_max, 
                            cv=cv, 
                            n_jobs=n_jobs, 
                            objective=objective, 
                            verbose=verbose
    )
        
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True),
                                pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(_objective, n_trials=budget, show_progress_bar=verbose)

    return SimpleNamespace(study=study, 
                           best_params=study.best_params, 
                           best_value=study.best_trial.value
    )



