# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import traceback
from collections import OrderedDict
from functools import partial

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
    StratifiedKFold,
)
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted

from ._split import RepeatedStratifiedGroupKFold, StratifiedGroupKFold
from .metrics import cvrmse, nmbe

logger = logging.getLogger("cross-validation")


#######################################################################################
# Utilities
#######################################################################################


def create_groups(X, group_block):
    if group_block == "day":
        grouped = X.groupby([lambda x: x.year, lambda x: x.dayofyear])
    elif group_block == "week":
        grouped = X.groupby([lambda x: x.year, lambda x: x.isocalendar()[1]])
    elif group_block == "month":
        grouped = X.groupby([lambda x: x.year, lambda x: x.month])
    else:
        raise ValueError("`groups` can be either `day`, `week` or `month`.")

    groups = None
    for i, (_, group) in enumerate(grouped):
        groups = pd.concat([groups, pd.Series(i, index=group.index)])
    return groups.reindex(X.index)


def create_splits(
    X, *, n_splits, n_repeats=None, group_by=None, stratify_by=None, random_state=None
):
    if (group_by is not None) and (stratify_by is not None):
        group_by = create_groups(X, group_by)
        stratify_by = create_groups(X, stratify_by)

        if (n_repeats is not None) and (n_repeats > 1):
            cv = RepeatedStratifiedGroupKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        return partial(cv.split, y=stratify_by, groups=group_by)

    elif group_by is not None:
        group_by = create_groups(X, group_by)

        if (n_repeats is not None) and (n_repeats > 1):
            cv = GroupShuffleSplit(
                n_splits=n_splits * n_repeats,
                test_size=1 / n_splits,
                random_state=random_state,
            )
        else:
            cv = GroupKFold(n_splits=n_splits)
        return partial(cv.split, y=None, groups=group_by)

    elif stratify_by is not None:
        stratify_by = create_groups(X, stratify_by)

        if (n_repeats is not None) and (n_repeats > 1):
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=False,
                random_state=random_state,
            )
        return partial(cv.split, y=stratify_by, groups=None)

    else:
        if (n_repeats is not None) and (n_repeats > 1):
            cv = RepeatedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            cv = KFold(
                n_splits=n_splits,
                shuffle=False,
                random_state=random_state,
            )
        return partial(cv.split, y=None, groups=None)


def fit_and_score(
    estimator,
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    scorers,
    target_name="consumption",
    fit_params=None,
    return_estimator=False,
):
    """Fit estimator and compute scores for a given dataset split.

    Args:
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
        target_name : str, default='consumption'
            It is expected that both y and the predictions of the `estimator` are
            dataframes with a single column, the name of which is the one provided
            for `target_name`.
        fit_params : dict or None
            Parameters that will be passed to ``estimator.fit``.
        return_estimator : bool, default=False
            Whether to return the fitted estimator.
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
    """Estimator that evaluates metric(s) by cross-validation.

    Args:
        estimator : Any regressor with scikit-learn API (i.e. with fit and predict methods)
            The object to use to fit the data and evaluate the metrics.
        group_by : str {None, 'day', 'week'}, default='week'
            Parameter that defines what constitutes an indivisible group of data. The same
            group will not appear in two different folds. If `group_by='week'`, the cross
            validation process will consider the different weeks of the year as groups. If
            `group_by='day'`, the different days of the year will be considered as groups.
            If None, no groups will be considered.
        stratify_by : str {None, 'week', 'month'}, default='month'
            Parameter that defines if the cross validation process will stratify the folds.
            If `stratify_by='month'`, the folds will preserve the percentage of month
            occurrences across test sets.
        n_splits : int, default=3
            Number of folds. Must be at least 2.
        n_repeats : int (default=None)
            Number of times the cross-validation process needs to be repeated.
        target_name : str, default='consumption'
            It is expected that both y and the predictions of the `estimator` are
            dataframes with a single column, the name of which is the one provided
            for `target_name`.
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
                    "NMBE": lambda y_true, y_pred:
                        eensight.pipelines.model_selection.nmbe(
                            y_true[target_name], y_pred[target_name]
                        )
                }
            )`
        keep_estimators : bool, default=False
            Whether to keep the fitted estimators per fold.
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
        random_state : int, RandomState instance or None (default=None)
            Controls the randomness of each repeated cross-validation instance.
            Pass an int for reproducible output across multiple function calls.
    """

    def __init__(
        self,
        estimator,
        group_by="week",
        stratify_by="month",
        n_splits=3,
        n_repeats=None,
        target_name="consumption",
        scorers=None,
        keep_estimators=False,
        n_jobs=None,
        verbose=True,
        fit_params=None,
        pre_dispatch="2*n_jobs",
        random_state=None,
    ):
        if (not hasattr(estimator, "fit")) or (not hasattr(estimator, "predict")):
            raise ValueError(
                "Invalid estimator. Please provide an estimator with `fit` and `predict` methods."
            )

        if group_by not in (None, "day", "week"):
            raise ValueError("`group_by` should be one of 'day', 'week'")

        if stratify_by not in (None, "week", "month"):
            raise ValueError("`stratify_by` should be one of 'week', 'month'")

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

    @property
    def n_parameters(self):
        check_is_fitted(self, "fitted_")
        n_parameters = 0
        for i, est in enumerate(self.estimators_):
            n_parameters += est.n_parameters
        return math.ceil(n_parameters / (i + 1))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
            y : pd.DataFrame, shape (n_samples, 1)
                The target dataframe.

        Returns:
            self : object
                Returns self.
        """
        try:
            check_is_fitted(self, "scores_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. " "Instantiate a new object."
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
            key: np.asarray([item[key] for item in results]).flatten()
            for key in results[0]
        }

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, include_components=False):
        """
        Args:
            X : pd.DataFrame, shape (n_samples, n_features)
                The input dataframe.
        """
        check_is_fitted(self, "fitted_")

        try:
            self.estimators_
        except AttributeError as exc:
            raise ValueError(
                "Prediction is possible only if the model is fitted "
                "with `keep_estimators=True`"
            ) from exc

        parallel = Parallel(n_jobs=self.n_jobs)
        results = parallel(
            delayed(est.predict)(X, include_components=include_components)
            for est in self.estimators_
        )

        prediction = None
        for i, result in enumerate(results):
            prediction = result if prediction is None else prediction + result

        return prediction / (i + 1)
