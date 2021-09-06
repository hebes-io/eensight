# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

from .cross_validation import create_groups
from .metrics import cvrmse, nmbe


def optimize(
    estimator,
    X,
    y,
    *,
    n_repeats=2,
    test_size=0.2,
    target_name="consumption",
    budget=20,
    timeout=None,
    scorers=None,
    directions=None,
    optimization_space=None,
    multivariate=False,
    out_of_sample=True,
    verbose=False,
    tags=None,
    **opt_space_kwards
):
    """
    Optimize a model's hyperparameters

    Args:
        estimator : Any regressor with scikit-learn API (i.e. with fit and
            predict methods) 
            The object to use to fit the data.
        X : pandas dataframe of shape (n_samples, n_features)
            The input data to optimize on.
        y : pandas dataframe of shape of shape (n_samples, 1)
            The training target data to optimize on.
        n_repeats : int, default=2
            Number of times to repeat the train/test data split process process.
        test_size : float, default=0.2
            The proportion of the dataset to include in the test split. Should be between
            0.0 and 1.0.
        target_name : str, default='consumption'
            It is expected that both y and the predictions of the `estimator` are
            dataframes with a single column, the name of which is the one provided
            for `target_name`.
        budget : int, default=20
            The number of trials. If this argument is set to `None`, there is no
            limitation on the number of trials. If `timeout` is also set to `None`,
            the study continues to create trials until it receives a termination
            signal such as Ctrl+C or SIGTERM.
        timeout : int, default=None
            Stop study after the given number of second(s). If this argument is set to
            `None`, the study is executed without time limitation.
        scorers : dict, default=None
            dict mapping scorer name to a callable. The callable object
            should have signature ``scorer(y_true, y_pred)``.
            The default value is:
            `OrderedDict(
                {
                    "CVRMSE": lambda y_true, y_pred:
                        eensight.pipelines.model_selection.cvrmse(
                            y_true[target_name], y_pred[target_name]
                        ),
                    "AbsNMBE": lambda y_true, y_pred: np.abs(
                        eensight.pipelines.model_selection.nmbe(
                            y_true[target_name], y_pred[target_name]
                        )
                    ),
                }
            )`
        directions : list, default=None
            A sequence of directions during multi-objective optimization. Set
            ``minimize`` for minimization and ``maximize`` for maximization.
            The default value is ['minimize', 'minimize'].
        optimization_space : callable, default=None
            A function that takes an `optuna.trial.Trial` as input and returns
            a parameter combination to try. If it is None, the `estimator` should
            have an `optimization_space` function.
        multivariate: bool, default=False
            If `True`, the multivariate TPE (Tree-structured Parzen Estimator)
            is used when suggesting parameters.
        out_of_sample: bool, default=True
            Whether the optimization should be based on out-of-sample (if `True`) or
            in-sample (if `False`) performance.
        verbose : bool, default=False
            Flag to show progress bars or not.
        tags: str or list of str, default=None
            Tags are returned by the function as-is and are useful as a way to distinguish
            the results when running the function many times in parallel.
        opt_space_kwards : dict
            Additional keyworded parameters to pass to the `optimization_space`
            function.
    """
    if (not hasattr(estimator, "fit")) or (not hasattr(estimator, "predict")):
        raise ValueError(
            "Invalid estimator. Please provide an estimator with `fit` and `predict` methods."
        )
    if (optimization_space is None) and (not hasattr(estimator, "optimization_space")):
        raise ValueError(
            "`optimization_space` must be provided either directly or by the estimator"
        )

    opt_space_fun = optimization_space or estimator.optimization_space
    if opt_space_kwards:
        opt_space_fun = functools.partial(opt_space_fun, **opt_space_kwards)

    if scorers is None:
        scorers = OrderedDict(
            {
                "CVRMSE": lambda y_true, y_pred: cvrmse(
                    y_true[target_name], y_pred[target_name]
                ),
                "AbsNMBE": lambda y_true, y_pred: np.abs(
                    nmbe(y_true[target_name], y_pred[target_name])
                ),
            }
        )
        directions = ["minimize", "minimize"]
    else:
        scorers = OrderedDict(scorers)

    if directions is None:
        directions = len(scorers) * ["minimize"]

    days = create_groups(X, "day").to_frame("day")
    months = create_groups(X, "month").to_frame("month")
    groups = pd.concat((days, months), axis=1)
    to_split = groups.groupby("day")["month"].first()

    def _objective(trial):
        param_space = opt_space_fun(trial)
        scores = np.zeros(len(scorers))

        if out_of_sample:
            for _ in range(n_repeats):
                train_index, test_index = train_test_split(
                    to_split.index,
                    test_size=test_size,
                    shuffle=True,
                    stratify=to_split.values,
                )
                train_index = groups.loc[groups["day"].isin(train_index)].index
                test_index = groups.loc[groups["day"].isin(test_index)].index
                X_train, y_train = X.loc[train_index], y.loc[train_index]
                X_test, y_test = X.loc[test_index], y.loc[test_index]

                estimator_cloned = clone(estimator)
                if hasattr(estimator_cloned, "apply_optimal_params"):
                    estimator_cloned.apply_optimal_params(**param_space)
                else:
                    estimator_cloned.set_params(**param_space)

                estimator_cloned = estimator_cloned.fit(X_train, y_train)
                pred = estimator_cloned.predict(X_test)

                for i, scorer in enumerate(scorers.values()):
                    score = scorer(y_test, pred)
                    scores[i] += score
            scores = scores / n_repeats
        else:
            estimator_cloned = clone(estimator)
            estimator_cloned.apply_optimal_params(**param_space)
            estimator_cloned = estimator_cloned.fit(X, y)
            pred = estimator_cloned.predict(X)

            for i, scorer in enumerate(scorers.values()):
                score = scorer(y, pred)
                scores[i] += score

        return tuple(scores)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=multivariate),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        directions=directions,
    )

    study.optimize(
        _objective, n_trials=budget, timeout=timeout, show_progress_bar=verbose
    )
    scores = defaultdict(list)
    params = defaultdict(list)
    for trial in study.best_trials:
        for name, value in zip(scorers, trial.values):
            scores[name].append(value)
        for name, value in trial.params.items():
            params[name].append(value)

    return Bunch(
        scores=pd.DataFrame.from_dict(scores),
        params=pd.DataFrame.from_dict(params),
        tags=tags,
    )
