# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from metric_learn import MMC
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from eensight.features.generate import DatetimeFeatures, MMCFeatures


def create_mmc_pairs(distances, pairs_per_prototype=100):
    daily_index = distances.index.map(lambda x: x.date)
    n_pairs = int(pairs_per_prototype * distances.shape[1])

    positive_pairs = np.ones((n_pairs, 3), dtype=np.int32)
    negative_pairs = (-1) * np.ones((n_pairs, 3), dtype=np.int32)

    for i, prototype in enumerate(distances.columns):
        threshold_low = distances[prototype].quantile(0.1)
        threshold_high = distances[prototype].quantile(0.8)

        start = pairs_per_prototype * i
        end = pairs_per_prototype * i + pairs_per_prototype

        for j in range(start, end):
            similar = distances[distances[prototype] <= threshold_low]
            similar = similar.sample(n=3, replace=False)
            similar = similar.sort_values(by=prototype)
            dissimilar = distances[distances[prototype] >= threshold_high].sample(n=1)

            positive_pairs[j, 0] = daily_index.get_loc(similar.index[0].date())
            positive_pairs[j, 1] = daily_index.get_loc(similar.index[1].date())
            negative_pairs[j, 0] = daily_index.get_loc(similar.index[2].date())
            negative_pairs[j, 1] = daily_index.get_loc(dissimilar.index[0].date())

    positive_pairs = np.unique(positive_pairs, axis=0)
    negative_pairs = np.unique(negative_pairs, axis=0)
    pairs = np.concatenate((positive_pairs, negative_pairs))
    return pairs


def learn_distance_metric(
    distances,
    pairs_per_prototype=100,
    test_size=0.5,
    return_features=False,
    return_pairs=False,
):
    feature_pipeline = Pipeline(
        [
            ("dates", DatetimeFeatures(subset=["month", "dayofweek"])),
            ("features", MMCFeatures()),
        ]
    )

    features = feature_pipeline.fit_transform(distances)
    pairs = create_mmc_pairs(distances, pairs_per_prototype=pairs_per_prototype)

    X_train, X_test, y_train, y_test = train_test_split(
        pairs[:, :2],
        pairs[:, -1],
        shuffle=True,
        stratify=pairs[:, -1],
        test_size=test_size,
    )

    mmc = MMC(preprocessor=np.array(features, dtype=float))
    mmc = mmc.fit(X_train, y_train)
    score = matthews_corrcoef(y_test, mmc.predict(X_test))
    return Bunch(
        score=score,
        metric_components=mmc.components_.transpose(),
        features=None if not return_features else features,
        pairs=None if not return_pairs else pairs,
    )


def metric_function(components, u, v, squared=False):
    """This function computes the metric between u and v, according to a
    learned metric.

    Parameters
    ----------
    components : numpy.ndarray
      The linear transformation `deduced from the learned Mahalanobis
      metric
    u : array-like, shape=(n_features,)
      The first point involved in the distance computation.
    v : array-like, shape=(n_features,)
      The second point involved in the distance computation.
    squared : bool
      If True, the function will return the squared metric between u and
      v, which is faster to compute.

    Returns
    -------
    distance : float
      The distance between u and v according to the new metric.
    """
    transformed_diff = (u - v).dot(components)
    dist = np.dot(transformed_diff, transformed_diff.T)
    if not squared:
        dist = np.sqrt(dist)
    return dist
