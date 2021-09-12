# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import functools
from collections import defaultdict
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from eensight.features.cluster import ClusterFeatures
from eensight.models.grouped import GroupedPredictor
from eensight.pipelines.day_typing.metric_learning import metric_function


class CompositePredictor(RegressorMixin, BaseEstimator):
    """Linear regression model that combines a clusterer (an estimator that answers
    the question "To which cluster should I allocate a given observation's target?")
    and a grouped regressor (regressor for predicting the target given information
    about the clusters)

    Args:
        distance_metrics : dict
            Dictionary containing time interval information of the form:
            key: interval start time, values: interval end time, and components of
            the corresponding distance metric.
        base_clusterer : eensight.features.cluster.ClusterFeatures
            An estimator that answers the question "To which cluster should I allocate
            a given observation's target?".
        base_regressor : eensight.models.grouped.GroupedPredictor
            A regressor for predicting the target given information about the clusters.
        group_feature : str, default='cluster'
            The name of the feature to use as the grouping set.
    """

    def __init__(
        self,
        *,
        distance_metrics: Dict,
        base_clusterer: ClusterFeatures,
        base_regressor: GroupedPredictor,
        group_feature="cluster",
    ):
        self.distance_metrics = distance_metrics
        self.base_clusterer = base_clusterer
        self.base_regressor = base_regressor
        self.group_feature = group_feature
        self.cluster_params_ = defaultdict(dict)

    @property
    def n_parameters(self):
        return self.estimator_.n_parameters

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
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

        X = X.assign(**{self.group_feature: -1})

        self.clusterers_ = {}
        for i, (start_time, props) in enumerate(self.distance_metrics.items()):
            end_time = (
                props["end_time"]
                if props["end_time"] is not None
                else datetime.time(0, 0)
            )
            subset = X.between_time(start_time, end_time, include_end=False)
            metric = functools.partial(metric_function, props["metric_components"])

            clusterer = clone(self.base_clusterer).set_params(
                assign_clusters__metric=metric, **self.cluster_params_[str(i)]
            )

            self.clusterers_[str(i)] = clusterer.fit(subset)
            clusters = clusterer.transform(subset)
            clusters[self.group_feature] = (
                clusters[self.group_feature]
                .astype(int)
                .astype(str)
                .map(lambda x: ":".join((str(i), x)))
            )
            X.loc[clusters.index, self.group_feature] = clusters[self.group_feature]

        self.estimator_ = clone(self.base_regressor)
        self.estimator_.fit(X, y)
        self.fitted_ = True
        return self

    def predict(
        self, X: pd.DataFrame, include_clusters=False, include_components=False
    ):
        check_is_fitted(self, "fitted_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input values are expected as pandas DataFrames.")
        X = X.assign(**{self.group_feature: -1})

        for i, (start_time, props) in enumerate(self.distance_metrics.items()):
            end_time = (
                props["end_time"]
                if props["end_time"] is not None
                else datetime.time(0, 0)
            )
            subset = X.between_time(start_time, end_time, include_end=False)
            clusterer = self.clusterers_[str(i)]
            clusters = clusterer.transform(subset)
            clusters[self.group_feature] = (
                clusters[self.group_feature]
                .astype(int)
                .astype(str)
                .map(lambda x: ":".join((str(i), x)))
            )
            X.loc[clusters.index, self.group_feature] = clusters[self.group_feature]

        return self.estimator_.predict(
            X, include_clusters=include_clusters, include_components=include_components
        )

    def optimization_space(self, trial, **kwargs):
        cluster_size_lower = kwargs.get("cluster_size_lower") or 7  # one week
        cluster_size_upper = kwargs.get("cluster_size_upper") or 60  # two months

        param_space = {
            f"{interval}__delim__assign_clusters__min_cluster_size": trial.suggest_int(
                f"{interval}__delim__assign_clusters__min_cluster_size",
                cluster_size_lower,
                cluster_size_upper,
            )
            for interval in range(len(self.distance_metrics))
        }
        return param_space

    def apply_optimal_params(self, **param_space):
        intervals = range(len(self.distance_metrics))

        for key, value in param_space.items():
            if "__delim__" in key:
                interval, param = key.split("__delim__")
                self.cluster_params_[interval].update({param: value})
            else:
                for interval in intervals:
                    self.cluster_params_[interval].update({key: value})
