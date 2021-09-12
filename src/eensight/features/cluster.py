# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from joblib import Memory
from sklearn.base import TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from eensight.base import BaseHeterogeneousEnsemble
from eensight.utils import check_X

NOISE = -1


class ClusterFeatures(TransformerMixin, BaseHeterogeneousEnsemble):
    """
    A composite transformer model that uses HDBSCAN (Hierarchical Density-Based
    Spatial Clustering of Applications with Noise) to cluster the input data and
    a KNeighborsClassifier to predict the clusters for unseen inputs.

    Args:
        min_cluster_size : int, optional (default=5)
            The minimum size of clusters; single linkage splits that contain
            fewer points than this will be considered points "falling out" of a
            cluster rather than a cluster splitting into two new clusters.
        min_samples : int, optional (default=5)
            The number of samples in a neighbourhood for a point to be
            considered a core point. This parameter controls what the clusterer
            identifies as noise.
        metric : string, or callable, optional (default='euclidean')
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by metrics.pairwise.pairwise_distances for its
            metric parameter. If metric is "precomputed", X is assumed to be a
            distance matrix and must be square.
        transformer : An object that implements a `fit_transform` method (default=None)
            The `fit_transform` method is used for transforming the input into a form
            that is understood by the distance metric.
        memory : Instance of joblib.Memory or string (optional)
            Used to cache the output of the computation of the tree.
            If a string is given, it is the path to the caching directory.
        allow_single_cluster : bool, optional (default=True)
            By default HDBSCAN* will not produce a single cluster, setting this
            to True will override this and allow single cluster results in
            the case that you feel this is a valid result for your dataset.
        cluster_selection_method : string, optional (default='eom')
            The method used to select clusters from the condensed tree. The
            standard approach for HDBSCAN* is to use an Excess of Mass algorithm
            to find the most persistent clusters. Alternatively you can instead
            select the clusters at the leaves of the tree -- this provides the
            most fine grained and homogeneous clusters. Options are:
                * ``eom``
                * ``leaf``
        n_neighbors : int, default=1
            Number of neighbors to use by default for :meth:`kneighbors` queries.
        weights : {'uniform', 'distance'} or callable, default='uniform'
            weight function used in prediction. Possible values:
            - 'uniform' : uniform weights.  All points in each neighborhood
            are weighted equally.
            - 'distance' : weight points by the inverse of their distance.
            In this case, closer neighbors of a query point will have a
            greater influence than neighbors which are further away.
            - [callable] : a user-defined function which accepts an
            array of distances, and returns an array of the same shape
            containing the weights.
        output_name : str, default='cluster'
            The name of the output dataframe's column that includes the cluster
            information.
    """

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=5,
        metric="euclidean",
        transformer=None,
        memory=Memory(None, verbose=0),
        allow_single_cluster=True,
        cluster_selection_method="eom",
        n_neighbors=1,
        weights="uniform",
        output_name="cluster",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.transformer = transformer
        self.memory = memory
        self.allow_single_cluster = allow_single_cluster
        self.cluster_selection_method = cluster_selection_method
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.output_name = output_name

        super().__init__(
            estimators=[
                (
                    "assign_clusters",
                    HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric=metric,
                        memory=memory,
                        allow_single_cluster=allow_single_cluster,
                        cluster_selection_method=cluster_selection_method,
                    ),
                ),
                (
                    "predict_clusters",
                    KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
                ),
            ],
        )

    def fit(self, X: pd.DataFrame, y=None):
        if self.transformer is not None:
            X = self.transformer.fit_transform(X)

        X = check_X(X)
        dt = X.index.to_series().diff()
        time_step = dt.iloc[dt.values.nonzero()[0]].min()

        if time_step < pd.Timedelta(days=1):
            X = X.groupby(lambda x: x.date).first()
            X.index = X.index.map(pd.to_datetime)

        clusterer = self.named_estimators["assign_clusters"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            clusterer.fit(X)

        self.clusters_ = pd.DataFrame(
            data=clusterer.labels_,
            index=X.index,
            columns=[self.output_name],
        )
        self.year_coverage_ = len(np.unique(X.index.dayofyear)) / 365

        classifier = self.named_estimators["predict_clusters"]
        without_noise = self.clusters_[self.clusters_[self.output_name] != NOISE].index

        if len(without_noise) == 0:
            classifier.fit(np.array(X), np.zeros(len(X)))
        else:
            classifier.fit(
                np.array(X.loc[without_noise]),
                np.array(self.clusters_.loc[without_noise, self.output_name]),
            )
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
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

        pred = pd.DataFrame(
            data=self.named_estimators["predict_clusters"].predict(X),
            index=X.index,
            columns=[self.output_name],
        )

        if index is not None:
            out = pd.Series(np.nan, index=index)
            for dt, grouped in pred.groupby(lambda x: x):
                out = out.mask(out.index.date == dt, grouped[self.output_name].item())
            pred = out.to_frame(self.output_name)

        return pred
