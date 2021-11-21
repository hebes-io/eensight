# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
import traceback
from collections import OrderedDict
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted

from eensight.utils import create_groups

from ._split import RepeatedStratifiedGroupKFold
from .metrics import cvrmse, nmbe

logger = logging.getLogger("cross-validation")


#######################################################################################
# Utilities
#######################################################################################


def create_splits(
    X, *, n_splits, n_repeats=None, group_by=None, stratify_by=None, random_state=None
):
    if (group_by is None) and (stratify_by is None):
        if (n_repeats is not None) and (n_repeats > 1):
            folds = RepeatedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            folds = KFold(n_splits=n_splits, shuffle=False)
        splits = partial(folds.split, y=None, groups=None)

    elif (group_by is not None) and (stratify_by is None):
        if (n_repeats is not None) and (n_repeats > 1):
            folds = GroupShuffleSplit(
                n_splits=n_splits * n_repeats,
                test_size=1 / n_splits,
                random_state=random_state,
            )
        else:
            folds = GroupKFold(n_splits=n_splits)
        splits = partial(folds.split, y=None, groups=create_groups(X, group_by))

    elif (group_by is None) and (stratify_by is not None):
        if (n_repeats is not None) and (n_repeats > 1):
            folds = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            folds = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        splits = partial(folds.split, y=create_groups(X, stratify_by), groups=None)

    else:
        if (n_repeats is not None) and (n_repeats > 1):
            folds = RepeatedStratifiedGroupKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            folds = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        splits = partial(
            folds.split,
            y=create_groups(X, stratify_by),
            groups=create_groups(X, group_by),
        )

    return splits


def fit_and_score(
    estimator: BaseEstimator,
    *,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    scorers: dict,
    target_name: str = "consumption",
    fit_params: dict = None,
    return_estimator: bool = False,
):
    """Fit estimator and compute scores for a given dataset split.

    Args:
        estimator (BaseEstimator): Estimator object implementing `fit` and
            `predict` methods.
        X_train (pandas.DataFrame): The training feature data to fit.
        y_train (pandas.DataFrame): The training target data to fit.
        X_test (pandas.DataFrame): The evaluation feature data for out-of-sample
            prediction.
        y_test (pandas.DataFrame): The testing target data.
        scorers (dict): A dictionary mapping scorer name to a callable. The callable
            object should have signature ``scorer(y_true, y_pred)``.
        target_name (str, optional): It is expected that both `y` and the predictions
            of the `estimator` are dataframes with a single column, the name of which
            is the one provided for `target_name`. Defaults to "consumption".
        fit_params (dict, optional): Parameters that will be passed to ``estimator.fit``.
            Defaults to None.
        return_estimator (bool, optional): Whether to return the fitted estimator.
            Defaults to True.

    Returns:
        dict: Dictionary containing the scorer names and scores, and the fitted estimator
            if `return_estimator` is True.
    """
    fit_params = fit_params if fit_params is not None else {}

    try:
        estimator.fit(X_train, y_train, **fit_params)
    except Exception as e:
        logger.warn(
            f"Estimator fit failed: {traceback.format_exc().splitlines()[-1]}. "
            "The score on this train-test partition for these parameters will "
            f"be set to {np.nan}. "
        )
        result = {name: np.nan for name in scorers}

        if return_estimator:
            result["estimator"] = None
    else:
        y_pred = estimator.predict(X_test)
        y_true = y_test.loc[y_pred.index]

        result = {
            name: np.array(scorer(y_true[target_name], y_pred[target_name]))
            for name, scorer in scorers.items()
        }

        if return_estimator:
            result["estimator"] = estimator

    return result


#######################################################################################
# CrossValidator
#######################################################################################


class CrossValidator(BaseEstimator):
    """Estimator that evaluates metrics by cross-validation.

    Args:
        estimator (BaseEstimator): Any regressor with scikit-learn API (i.e. with fit
            and predict methods). The estimator to fit the data and evaluate the metrics.
        group_by ({None, 'day', 'week'}, optional): Parameter that defines what constitutes
            an indivisible group of data. The same group will not appear in two different
            folds. If `group_by='week'`, the cross validation process will consider the
            different weeks of the year as groups. If `group_by='day'`, the different days
            of the year will be considered as groups. If None, no groups will be considered.
            Defaults to "week".
        stratify_by ({None, 'week', 'month'}, optional): Parameter that defines if the cross
            validation process will stratify the folds. For example, If `stratify_by='month'`,
            the folds will preserve the percentage of month occurrences across test sets.
            Defaults to "month".
        n_splits (int, optional): Number of folds. Must be at least 2. Defaults to 3.
        n_repeats (int, optional): Number of times the cross-validation process needs to be
            repeated. Defaults to None.
        target_name (str, optional): It is expected that both target `y` and the predictions
            of the `estimator` are dataframes with a single column, the name of which is the
            one provided for `target_name`. Defaults to "consumption".
        scorers (dict, optional): A dictionary mapping scorer name to a callable. The callable
            object should have signature ``scorer(y_true, y_pred)``. The default value is:
            ::
                OrderedDict(
                    {
                        "CVRMSE": lambda y_true, y_pred:
                            eensight.pipelines.baseline.cvrmse(
                                y_true[target_name], y_pred[target_name]
                            ),
                        "NMBE": lambda y_true, y_pred:
                            eensight.pipelines.baseline.nmbe(
                                y_true[target_name], y_pred[target_name]
                            )
                    }
                )
            Defaults to None.
        keep_estimators (bool, optional): Whether to keep the fitted estimators per fold.
            Defaults to False.
        n_jobs (int, optional): Number of jobs to run in parallel. Training the estimator
            and computing the score are parallelized over the cross-validation splits.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. Defaults to None.
        verbose (bool, optional): The verbosity level. Defaults to True.
        fit_params (dict, optional): Parameters to pass to the `fit` method of the estimator.
            Defaults to None.
        pre_dispatch (Union[int, str], optional): Controls the number of jobs that get
            dispatched during parallel execution. Reducing this number can be useful to
            avoid an explosion of memory consumption when more jobs get dispatched than CPUs
            can process. This parameter can be:

                - None, in which case all the jobs are immediately created and spawned. Use
                this for lightweight and fast-running jobs, to avoid delays due to on-demand
                spawning of the jobs

                - An int, giving the exact number of total jobs that are spawned

                - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

            Defaults to "2*n_jobs".
        random_state (int or RandomState instance, optional): Controls the randomness of each
            repeated cross-validation instance. Pass an int for reproducible output across
            multiple function calls. Defaults to None.

    Raises:
        ValueError: If `estimator` does not have `fit` and `predict` methods.
        ValueError: If `group_by` is not one of 'day', 'week' or None.
        ValueError: If `stratify_by` is not one of 'week', 'month' or None.
        ValueError: If `group_by` is a longer period than `stratify_by`.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        group_by: str = "week",
        stratify_by: str = "month",
        n_splits: int = 3,
        n_repeats: int = None,
        target_name: str = "consumption",
        scorers: dict = None,
        keep_estimators: bool = False,
        n_jobs: int = None,
        verbose: bool = True,
        fit_params: dict = None,
        pre_dispatch: Union[int, str] = "2*n_jobs",
        random_state: int = None,
    ):
        if (not hasattr(estimator, "fit")) or (not hasattr(estimator, "predict")):
            raise ValueError(
                "Invalid estimator. Please provide an estimator with `fit` and `predict` methods."
            )

        if group_by not in (None, "day", "week"):
            raise ValueError("`group_by` should be one of 'day', 'week' or None")

        if stratify_by not in (None, "week", "month"):
            raise ValueError("`stratify_by` should be one of 'week', 'month' or None")

        if (
            (group_by is not None)
            and (stratify_by is not None)
            and ("day", "week").index(group_by) > ("week", "month").index(stratify_by)
        ):
            raise ValueError("`group_by` should be a shorter period than `stratify_by`")

        self.estimator = estimator
        self.group_by = group_by
        self.stratify_by = stratify_by
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.target_name = target_name
        self.scorers = scorers
        self.keep_estimators = keep_estimators
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fit_params = fit_params
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state

    @property
    def estimators(self):
        try:
            self.estimators_
        except AttributeError as exc:
            raise ValueError(
                "`estimators` are available only if the model is fitted "
                "with `keep_estimators=True`"
            ) from exc
        else:
            return self.estimators_

    @property
    def oos_masks(self):
        try:
            self.estimator_oos_masks_
        except AttributeError as exc:
            raise ValueError(
                "`oos_masks` are available only if the model is fitted "
                "with `keep_estimators=True`"
            ) from exc
        else:
            return self.estimator_oos_masks_

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the cross-validator on the available data.

        Args:
            X (pandas.DataFrame): The input dataframe.
            y (pandas.DataFrame): The target dataframe.

        Returns:
            CrossValidator: Fitted cross-validator.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        if self.scorers is None:
            self.scorers = OrderedDict({"CVRMSE": cvrmse, "NMBE": nmbe})

        splits = create_splits(
            X,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            group_by=self.group_by,
            stratify_by=self.stratify_by,
            random_state=self.random_state,
        )

        cv_indices = []
        for train_idx, test_idx in splits(X):
            cv_indices.append((train_idx, test_idx))

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch
        )
        results = parallel(
            delayed(fit_and_score)(
                clone(self.estimator),
                X_train=X.iloc[train],
                y_train=y.iloc[train],
                X_test=X.iloc[test],
                y_test=y.iloc[test],
                scorers=self.scorers,
                target_name=self.target_name,
                fit_params=self.fit_params,
                return_estimator=self.keep_estimators,
            )
            for (train, test) in cv_indices
        )

        if self.keep_estimators:
            self.estimators_ = []
            self.estimator_oos_masks_ = []
            for item, (_, test) in zip(results, cv_indices):
                estimator = item.pop("estimator")
                if estimator is not None:
                    self.estimators_.append(estimator)
                    self.estimator_oos_masks_.append(X.index[test])

        self.scores_ = {
            key: np.asarray(
                [item[key] for item in results if item is not None]
            ).flatten()
            for key in results[0]
        }

        self.fitted_ = True
        return self
