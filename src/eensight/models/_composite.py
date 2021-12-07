# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import datetime
import functools
import logging
import math
import multiprocessing
import warnings
from collections import OrderedDict, defaultdict
from typing import Callable, Union

import numpy as np
import optuna
import pandas as pd
from feature_encoders.models import GroupedPredictor
from feature_encoders.utils import as_list
from joblib import Parallel
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)
logger = logging.getLogger("model-training")

from eensight.features import ClusterFeatures
from eensight.metrics import cvrmse
from eensight.pipelines.daytype import metric_function
from eensight.utils import create_groups


class CompositePredictor(RegressorMixin, BaseEstimator):
    """Linear regression model that combines a clusterer and a regressor.

    Args:
        distance_metric (string or callable): The metric to use when calculating
            distance between instances in a feature array. The metric will be
            passed to the ``base_clusterer``.
        base_clusterer (eensight.features.cluster.ClusterFeatures): An estimator
            that answers the question "To which cluster should I allocate a given
            observation's target?".
        base_regressor (feature_encoders.models.GroupedPredictor): A regressor for
            predicting the target given information about the clusters.
        cluster_params (dict, optional): The parameters of the `base_clusterer`.
            Defaults to `defaultdict(dict)`.
        group_feature (str, optional): The name of the feature to use as the grouping
            set. Defaults to 'cluster'.
    """

    def __init__(
        self,
        *,
        distance_metric: Union[str, Callable],
        base_clusterer: ClusterFeatures,
        base_regressor: GroupedPredictor,
        cluster_params: dict = defaultdict(dict),
        group_feature: str = "cluster",
    ):
        self.distance_metric = distance_metric
        self.base_clusterer = base_clusterer
        self.base_regressor = base_regressor
        self.cluster_params = cluster_params
        self.group_feature = group_feature

    @property
    def n_parameters(self):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError as exc:
            raise ValueError(
                "The number of parameters is acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            n_parameters = 0
            for i, estimator in enumerate(self.estimators_):
                n_parameters += estimator["regressor"].n_parameters
            return math.ceil(n_parameters / (i + 1))

    @property
    def dof(self):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError as exc:
            raise ValueError(
                "The degrees of freedom are acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            dof = 0
            for i, estimator in enumerate(self.estimators_):
                dof += estimator["regressor"].dof
            return math.ceil(dof / (i + 1))

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        """Fit the estimator with the available data.

        Args:
            X (pandas.DataFrame): Input data.
            y (pandas.Series or pandas.DataFrame): Target data.

        Raises:
            Exception: If the estimator is re-fitted. An estimator object can
                only be fitted once.
            ValueError: If input `X` is not a pandas DataFrame.
            ValueError: If the name of ``group_feature`` alreday exists in input's
                columns.

        Returns:
            CompositePredictor: Fitted estimator.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input values are expected as pandas DataFrames.")
        if self.group_feature in X.columns:
            raise ValueError(
                f"Name `{self.group_feature}` already registered as grouping set name"
            )

        self.estimators_ = []
        if len(self.cluster_params) == 0:
            logger.warning(
                "Model is being fitted without explicit clustering parameters"
            )
            clusterer = clone(self.base_clusterer).set_params(
                assign_clusters__metric=self.distance_metric
            )
            clusters = clusterer.fit_transform(X)
            X_with_clusters = pd.concat((X, clusters), axis=1)
            regressor = clone(self.base_regressor).fit(X_with_clusters, y)
            self.estimators_.append({"clusterer": clusterer, "regressor": regressor})
        else:
            for params in self.cluster_params.values():
                clusterer = clone(self.base_clusterer).set_params(
                    assign_clusters__metric=self.distance_metric,
                    **params,
                )
                clusters = clusterer.fit_transform(X)
                X_with_clusters = pd.concat((X, clusters), axis=1)
                regressor = clone(self.base_regressor).fit(X_with_clusters, y)
                self.estimators_.append(
                    {"clusterer": clusterer, "regressor": regressor}
                )
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, include_components=False):
        """Predict given new input data.

        Args:
            X (pandas.DataFrame): Input data.
            include_components (bool, optional): Whether to include the contribution of the
                individual components of the model structure in the returned prediction.
                Defaults to False.

        Raises:
            ValueError: If input `X` is not a pandas DataFrame.

        Returns:
            pandas.DataFrame: The predicted values.
        """
        check_is_fitted(self, "fitted_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input values are expected as pandas DataFrames.")

        prediction = 0
        for i, estimator in enumerate(self.estimators_):
            clusters = estimator["clusterer"].transform(X)
            X_with_clusters = pd.concat((X, clusters), axis=1)
            pred = estimator["regressor"].predict(
                X_with_clusters, include_components=include_components
            )
            prediction += pred

        return prediction / (i + 1)

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        n_estimators: int = 3,
        test_size: float = 0.33,
        group_by: str = "week",
        budget: int = 50,
        timeout: int = None,
        scorers: dict = None,
        directions: list = None,
        multivariate: bool = True,
        verbose: bool = False,
        **opt_space_kwards,
    ):
        """Optimize a model's hyperparameters.

        Args:
            X (pandas.DataFrame): The input data to optimize on.
            y (pandas.DataFrame): The training target data to optimize on.
            n_estimators (int, optional): The number of estimators to include. If
                `n_estimators=n`, then the `n` best parameters of the optimization
                will be used to create an ensemble of `n` pairs of `base_clusterer`
                and `base_regressor`. Defaults to 3.
            test_size (float, optional): The proportion of the dataset to include
                in the test split. Should be between 0.0 and 1.0. Defaults to 0.33.
            group_by ({'day', 'week'}, optional): Parameter that defines what constitutes
                an indivisible group of data. The same group will not appear in both train
                and test subsets. If `group_by='week'`, the optimization process will
                consider the different weeks of the year as groups. If `group_by='day'`,
                the different days of the year will be considered as groups.
                Defaults to "week".
            budget (int, optional): The number of trials. If this argument is set to `None`,
                there is no limitation on the number of trials. If `timeout` is also set to
                `None`, the study continues to create trials until it receives a termination
                signal such as Ctrl+C or SIGTERM. Defaults to 50.
            timeout (int, optional): Stop study after the given number of second(s). If this
                argument is set to `None`, the study is executed without time limitation.
                Defaults to None.
            scorers (dict, optional): Dictionary mapping scorer name to a callable. The
                callable object should have signature ``scorer(y_true, y_pred)``.
                Defaults to None.
            directions (list, optional): A sequence of directions during multi-objective
                optimization. Set ``['minimize']`` for minimization and ``['maximize']``
                for maximization. The default value is ``['minimize']``.
                Defaults to None.
            multivariate (bool, optional): If `True`, the multivariate TPE (Tree-structured
                Parzen Estimator) is used when suggesting parameters. Defaults to True.
            verbose (bool, optional): Flag to show progress bars or not. Defaults to False.
            *opt_space_kwards: Additional keyworded parameters to use in the optimization space.

        Raises:
            ValueError: If `group_by` is neither "day" or "week".

        Returns:
            dict: The optimized parameters.
        """
        if group_by not in ("day", "week"):
            raise ValueError("Parameter `group_by` can be either `day` or `week`.")

        if scorers is None:
            scorers = {
                "CVRMSE": lambda y_true, y_pred: cvrmse(
                    y_true["consumption"], y_pred["consumption"]
                ),
            }
        else:
            scorers = OrderedDict(scorers)

        if directions is None:
            directions = len(scorers) * ["minimize"]
        else:
            directions = as_list(directions)

        epsilon_lower = opt_space_kwards.get("epsilon_lower") or 0.5
        epsilon_upper = opt_space_kwards.get("epsilon_upper") or 1.5
        min_samples_lower = opt_space_kwards.get("min_samples_lower") or 14
        min_samples_upper = opt_space_kwards.get("min_samples_upper") or 30

        def _objective(trial):
            param_space = {
                "assign_clusters__eps": trial.suggest_float(
                    "assign_clusters__eps",
                    epsilon_lower,
                    epsilon_upper,
                ),
                "assign_clusters__min_samples": trial.suggest_int(
                    "assign_clusters__min_samples",
                    min_samples_lower,
                    min_samples_upper,
                ),
            }
            scores = np.zeros(len(scorers))

            groups = create_groups(X, group_by)
            test = pd.Series(groups.unique()).sample(frac=test_size)
            test_index = groups.loc[groups.isin(test)].index

            X_train = X[~np.isin(X.index, test_index)]
            X_test = X[np.isin(X.index, test_index)]
            y_train = y[np.isin(y.index, X_train.index)]
            y_test = y[np.isin(y.index, X_test.index)]

            estimator_cloned = clone(self)
            estimator_cloned = estimator_cloned.set_params(
                cluster_params={
                    0: {
                        "assign_clusters__eps": float(
                            param_space["assign_clusters__eps"]
                        ),
                        "assign_clusters__min_samples": int(
                            param_space["assign_clusters__min_samples"]
                        ),
                    }
                }
            )

            estimator_cloned = estimator_cloned.fit(X_train, y_train)
            pred = estimator_cloned.predict(X_test)

            for i, scorer in enumerate(scorers.values()):
                score = scorer(y_test, pred)
                scores[i] = score
            return tuple(scores)

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(multivariate=multivariate),
            directions=directions,
        )
        study.optimize(
            _objective, n_trials=budget, timeout=timeout, show_progress_bar=verbose
        )

        selected = (
            study.trials_dataframe(attrs=("value", "params"))
            .sort_values(by="value", ascending=True)
            .drop("value", axis=1)
            .iloc[:n_estimators]
        )

        if len(selected) != n_estimators:
            raise ValueError(
                "The number of trials cannot be smaller than `n_estimators`"
            )

        selected.columns = selected.columns.map(lambda x: x.replace("params_", ""))

        cluster_params = {
            i: {
                "assign_clusters__eps": float(row["assign_clusters__eps"]),
                "assign_clusters__min_samples": int(
                    row["assign_clusters__min_samples"]
                ),
            }
            for i, (_, row) in enumerate(selected.iterrows())
        }
        return cluster_params


class AggregatePredictor(RegressorMixin, BaseEstimator):
    """Linear regression model that aggregates a clusterer and a regressor for different
    time intervals.

    Args:
        distance_metrics (dict): A dictionary that contains time interval information of
            the form: key: interval number, values: interval start time, interval end time,
            and components of the corresponding distance metric.
        base_clusterer (eensight.features.cluster.ClusterFeatures): An estimator
            that answers the question "To which cluster should I allocate a given
            observation's target?".
        base_regressor (feature_encoders.models.GroupedPredictor): A regressor for
            predicting the target given information about the clusters.
        cluster_params (dict, optional): A dictionary that contains information of the form:
            key: interval number, values: the parameters of the corresponding `base_clusterer`.
            Defaults to `defaultdict(lambda: defaultdict(dict))`.
        group_feature (str, optional): The name of the feature to use as the grouping
            set. Defaults to 'cluster'.
    """

    def __init__(
        self,
        *,
        distance_metrics: dict,
        base_clusterer: ClusterFeatures,
        base_regressor: GroupedPredictor,
        cluster_params: dict = defaultdict(lambda: defaultdict(dict)),
        group_feature="cluster",
    ):
        self.distance_metrics = distance_metrics
        self.base_clusterer = base_clusterer
        self.base_regressor = base_regressor
        self.cluster_params = cluster_params
        self.group_feature = group_feature

        for props in self.distance_metrics.values():
            if props["end_time"] is None:
                props["end_time"] = (
                    datetime.datetime.combine(
                        datetime.datetime.now().date(),
                        self.distance_metrics[0]["start_time"],
                    )
                    - datetime.timedelta(seconds=1)
                ).time()

    @property
    def n_parameters(self):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError as exc:
            raise ValueError(
                "The number of parameters is acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            return sum([est.n_parameters for est in self.estimators_])

    @property
    def dof(self):
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError as exc:
            raise ValueError(
                "The degrees of freedom are acceccible only after "
                "the model has been fitted"
            ) from exc
        else:
            return sum([est.dof for est in self.estimators_])

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        """Fit the estimator with the available data.

        Args:
            X (pandas.DataFrame): Input data.
            y (pandas.Series or pandas.DataFrame): Target data.

        Raises:
            Exception: If the estimator is re-fitted. An estimator object can
                only be fitted once.
            ValueError: If input `X` is not a pandas DataFrame.

        Returns:
            EnsemblePredictor: Fitted estimator.
        """
        try:
            check_is_fitted(self, "fitted_")
        except NotFittedError:
            pass
        else:
            raise Exception(
                "Estimator object can only be fit once. Instantiate a new object."
            )

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input values are expected as pandas DataFrames.")

        self.estimators_ = []

        for i, props in self.distance_metrics.items():
            start_time = props["start_time"]
            end_time = props["end_time"]
            subset = X.between_time(start_time, end_time, include_end=False)
            cluster_params = self.cluster_params[i]
            estimator = CompositePredictor(
                distance_metric=functools.partial(
                    metric_function, props["metric_components"]
                ),
                base_clusterer=clone(self.base_clusterer),
                base_regressor=clone(self.base_regressor),
                cluster_params=cluster_params,
                group_feature=self.group_feature,
            )
            self.estimators_.append(estimator.fit(subset, y.loc[subset.index]))

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, include_components=False):
        """Predict given new input data.

        Args:
            X (pandas.DataFrame): Input data.
            include_components (bool, optional): Whether to include the contribution of the
                individual components of the model structure in the returned prediction.
                Defaults to False.

        Raises:
            ValueError: If input `X` is not a pandas DataFrame.

        Returns:
            pandas.DataFrame: The predicted values.
        """
        check_is_fitted(self, "fitted_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input values are expected as pandas DataFrames.")

        prediction = []
        for estimator, props in zip(self.estimators_, self.distance_metrics.values()):
            start_time = props["start_time"]
            end_time = props["end_time"]
            subset = X.between_time(start_time, end_time, include_end=False)
            prediction.append(
                estimator.predict(
                    subset,
                    include_components=include_components,
                )
            )
        prediction = pd.concat(prediction).reindex(X.index)
        return prediction

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        n_estimators: int = 3,
        test_size: float = 0.33,
        group_by: str = "week",
        budget: int = 50,
        timeout: int = None,
        scorers: dict = None,
        directions: list = None,
        multivariate: bool = True,
        n_jobs: int = None,
        verbose: bool = False,
        **opt_space_kwards,
    ):
        """Optimize a model's hyperparameters.

        Args:
            X (pandas.DataFrame): The input data to optimize on.
            y (pandas.DataFrame): The training target data to optimize on.
            n_estimators (int, optional): The number of estimators to include. If
                `n_estimators=n`, then the `n` best parameters of the optimization
                will be used to create an ensemble of `n` pairs of `base_clusterer`
                and `base_regressor`. Defaults to 3.
            test_size (float, optional): The proportion of the dataset to include
                in the test split. Should be between 0.0 and 1.0. Defaults to 0.33.
            group_by ({'day', 'week'}, optional): Parameter that defines what constitutes
                an indivisible group of data. The same group will not appear in both train
                and test subsets. If `group_by='week'`, the optimization process will
                consider the different weeks of the year as groups. If `group_by='day'`,
                the different days of the year will be considered as groups.
                Defaults to "week".
            budget (int, optional): The number of trials. If this argument is set to `None`,
                there is no limitation on the number of trials. If `timeout` is also set to
                `None`, the study continues to create trials until it receives a termination
                signal such as Ctrl+C or SIGTERM. Defaults to 50.
            timeout (int, optional): Stop study after the given number of second(s). If this
                argument is set to `None`, the study is executed without time limitation.
                Defaults to None.
            scorers (dict, optional): Dictionary mapping scorer name to a callable. The
                callable object should have signature ``scorer(y_true, y_pred)``.
                Defaults to None.
            directions (list, optional): A sequence of directions during multi-objective
                optimization. Set ``['minimize']`` for minimization and ``['maximize']``
                for maximization. The default value is ``['minimize']``.
                Defaults to None.
            multivariate (bool, optional): If `True`, the multivariate TPE (Tree-structured
                Parzen Estimator) is used when suggesting parameters. Defaults to True.
            n_jobs (int, optional): Number of jobs to run in parallel. ``None`` means 1 unless
                in a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
                Defaults to None.
            verbose (bool, optional): Flag to show progress bars or not. Defaults to False.
            *opt_space_kwards: Additional keyworded parameters to use in the optimization space.

        Raises:
            ValueError: If input `X` is not a pandas DataFrame.

        Returns:
            dict: The optimized parameters.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input values are expected as pandas DataFrames.")

        def _split(X, props):
            return X.between_time(
                props["start_time"], props["end_time"], include_end=False
            )

        if n_jobs == -1:
            n_cores = multiprocessing.cpu_count()
            n_jobs = min(n_cores, len(self.distance_metrics))

        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
        opt_params = parallel(
            delayed(
                CompositePredictor(
                    distance_metric=functools.partial(
                        metric_function, props["metric_components"]
                    ),
                    base_clusterer=clone(self.base_clusterer),
                    base_regressor=clone(self.base_regressor),
                    group_feature=self.group_feature,
                ).optimize
            )(
                _split(X, props),
                _split(y, props),
                n_estimators=n_estimators,
                test_size=test_size,
                group_by=group_by,
                budget=budget,
                timeout=timeout,
                scorers=scorers,
                directions=directions,
                multivariate=multivariate,
                verbose=verbose,
                **opt_space_kwards,
            )
            for props in self.distance_metrics.values()
        )
        return {
            key: params for key, params in zip(self.distance_metrics.keys(), opt_params)
        }
