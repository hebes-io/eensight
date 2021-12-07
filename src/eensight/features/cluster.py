# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from feature_encoders.utils import check_X
from sklearn.base import TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from eensight.base import BaseHeterogeneousEnsemble
from eensight.utils import recover_missing_dates

NOISE = -1


class ClusterFeatures(TransformerMixin, BaseHeterogeneousEnsemble):
    """Create cluster features.

    This is a composite transformer model that uses `DBSCAN` (Density-Based Spatial Clustering
    of Applications with Noise) to cluster the input data and a `KNeighborsClassifier` to assign
    clusters to unseen inputs.

    Args:
        eps (float, optional): The maximum distance between two samples for one to be
            considered as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. Defaults to 0.5.
        min_samples (int, optional): The number of samples in a neighbourhood for
            a point to be considered a core point. This parameter controls what the
            clusterer identifies as noise. Defaults to 5.
        metric (string or callable, optional): The metric to use when calculating
            distance between instances in a feature array. It must be one of the options
            allowed by metrics.pairwise.pairwise_distances for its metric parameter. Defaults
            to 'euclidean'.
        metric_params (dict, optional): Additional keyword arguments for the metric function.
            Defaults to None.
        transformer (sklearn.base.BaseEstimator, optional): An object that implements a
            `fit_transform` method, which is used for transforming the input into a form
            that is understood by the distance metric. Defaults to None.
        n_jobs (int, optional): The number of parallel jobs to run for the clusterer. ``None``
            means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors. Defaults to None.
        n_neighbors (int, optional): Number of neighbors to use by default for :meth:`kneighbors`
            queries. This parameter is passed to ``sklearn.neighbors.KNeighborsClassifier``.
            Defaults to 1.
        weights ({'uniform', 'distance'} or callable, optional): The weight function used in
            prediction. This parameter is passed to ``sklearn.neighbors.KNeighborsClassifier``.
            Defaults to 'uniform'.
        output_name (str, optional): The name of the output dataframe's column that includes
            the cluster information. Defaults to 'cluster'.
    """

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        transformer=None,
        n_jobs=None,
        n_neighbors=1,
        weights="uniform",
        output_name="cluster",
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.transformer = transformer
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.output_name = output_name

        super().__init__(
            estimators=[
                (
                    "assign_clusters",
                    DBSCAN(
                        eps=eps,
                        min_samples=min_samples,
                        metric=metric,
                        metric_params=metric_params,
                        n_jobs=n_jobs,
                    ),
                ),
                (
                    "predict_clusters",
                    KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
                ),
            ],
        )

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature generator on the available data.

        Args:
            X (pandas.DataFrame): The input dataframe.
            y (None, optional): Ignored. Defaults to None.

        Raises:
            ValueError: If the input data do not pass the checks of
                `feature_encoders.utils.check_X`.

        Returns:
            ClusterFeatures: The fitted instance.
        """
        if self.transformer is not None:
            X = self.transformer.fit_transform(X)

        X = check_X(X)
        dt = X.index.to_series().diff()
        time_step = dt.iloc[dt.values.nonzero()[0]].min()

        if time_step < pd.Timedelta(days=1):  # the clustering is applied on days
            X = X.groupby(lambda x: x.date).first()
            X.index = X.index.map(pd.to_datetime)

        clusterer = self.named_estimators["assign_clusters"]
        clusterer = clusterer.fit(X)

        self.clusters_ = pd.DataFrame(
            data=clusterer.labels_,
            index=X.index,
            columns=[self.output_name],
        )
        self.year_coverage_ = len(np.unique(X.index.dayofyear)) / 365

        classifier = self.named_estimators["predict_clusters"]
        without_noise_idx = self.clusters_[
            self.clusters_[self.output_name] != NOISE
        ].index

        if len(without_noise_idx) == 0:
            classifier.fit(np.array(X), np.zeros(len(X)))
        else:
            classifier.fit(
                np.array(X.loc[without_noise_idx]),
                np.array(self.clusters_.loc[without_noise_idx, self.output_name]),
            )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the feature generator.

        Args:
            X (pandas.DataFrame): The input dataframe.

        Returns:
            pandas.DataFrame: The transformed dataframe.
        """
        check_is_fitted(self, "fitted_")

        if self.transformer is not None:
            X = self.transformer.fit_transform(X)

        X = check_X(X)
        dt = X.index.to_series().diff()
        time_step = dt.iloc[dt.values.nonzero()[0]].min()

        index = None
        if time_step < pd.Timedelta(days=1):
            index = X.index
            X = X.groupby(lambda x: x.date).first()
            X.index = X.index.map(pd.to_datetime)

        pred = pd.DataFrame(
            data=self.named_estimators["predict_clusters"].predict(X),
            index=X.index,
            columns=[self.output_name],
        )

        if index is not None:
            idx_ext = recover_missing_dates(pd.DataFrame(index=index)).index
            pred.index = pred.index.map(lambda x: x.replace(hour=0, minute=0, second=0))
            pred = (
                idx_ext.to_frame().join(pred).fillna(method="ffill")[[self.output_name]]
            )
            pred = pred.reindex(index)

        return pred
