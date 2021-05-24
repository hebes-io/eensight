# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys 
import logging
import traceback

import numpy as np

from joblib import Parallel
from sklearn.base import clone 
from types import SimpleNamespace
from sklearn.utils.fixes import delayed
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as msqe


logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO, 
                    format='%(name)s %(asctime)s:%(levelname)-8s: %(message)s'
)
logger = logging.getLogger(__file__)



def fit_and_score(estimator, X_train, y_train, X_test, y_test, scorers, 
                    fit_params=None, return_estimator=False):
    """Fit estimator and compute scores for a given dataset split.
    
    Parameters
    ----------
    estimator : estimator object implementing a `fit` method
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
        object should have signature ``scorer(y_true, y_pred)``.
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
                    f'The score on this train-test partition will be set to {np.nan}. '
        )
        result = {
            name: np.nan for name in scorers
        }
    else:
        y_pred = estimator.predict(X_test)
        result = {
            name: np.array(scorer(y_test.loc[y_pred.index], y_pred)) 
                for name, scorer in scorers.items()
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
    estimator : estimator object implementing a `fit` method
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
        object should have signature ``scorer(y_true, y_pred)``.
    cv : cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
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
    if scorers is None:
        scorers = {
            'CVRMSE': lambda y_true, y_pred: msqe(y_true, y_pred, squared=False)/np.mean(y_true),
        }

    if cv is None:
        cv = KFold(n_splits=5, shuffle=False)
    
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(delayed(fit_and_score)(clone(estimator), X.iloc[train], y.iloc[train], 
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