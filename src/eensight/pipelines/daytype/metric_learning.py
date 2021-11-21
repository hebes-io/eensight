# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from feature_encoders.generate import DatetimeFeatures
from metric_learn import MMC
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from eensight.features import MMCFeatures


def create_mmc_pairs(distances: pd.DataFrame, pairs_per_prototype: int = 50):
    """Create positive and negative pairs of (sub)daily profiles for distance metric learning.

    The positive and negative pairs will be provided as input to a distance metric learning
    algorithm, the goal of which is to learn a distance metric that puts positive pairs
    close together and negative pairs far away. Positive pairs are constructed from (sub)daily
    profiles that are both similar to a given prototype, whereas negative pairs include one
    profile that is similar to a given prototype and one that is not.

    Args:
        distances (pandas.DataFrame): The Euclidean distances of all (sub)daily profiles from
            each prototype.
        pairs_per_prototype (int, optional): The number of pairs per prototype to generate.
            Defaults to 50.

    Returns:
        numpy array: An array of three columns. The first two correspond to the beginning index
            of each of the (sub)daily profiles in a pair, while the third is 1 if they are similar
            and -1 if they are not.
    """
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
    distances: pd.DataFrame,
    pairs_per_prototype: int = 50,
    test_size: float = 0.25,
    return_features: bool = False,
    return_pairs: bool = False,
):
    """Learn a distance metric given the distances from the prototypes.

    Args:
        distances (pandas.DataFrame): The Euclidean distances of all (sub)daily profiles
            from each prototype.
        pairs_per_prototype (int, optional): The number of pairs per prototype to generate.
            Defaults to 50.
        test_size (float, optional): The percentage of the pairs to keep for evaluation.
            For the evaluation, we test whether given the time information of two observations
            (day of week and month of year), the model can accurately predict whether the
            corresponding profiles are similar or not (evaluated as a binary classification
            problem). Defaults to 0.25.
        return_features (bool, optional): Whether to return the features that were used to train
            the distance metric model. Defaults to False.
        return_pairs (bool, optional): Whether to return the beginning indices of the pairs.
            Defaults to False.

    Returns:
        sklearn.utils.Bunch: A dict-like structure that includes the Matthews correlation
            coefficient of the evaluation of the test data, the numpy array of the linear
            transformation that was deduced from the learned Mahalanobis metric, the features
            that were used to train the distance metric model, and the beginning indices of the
            pairs.
    """
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


def metric_function(
    components: np.ndarray, u: np.ndarray, v: np.ndarray, squared: bool = False
):
    """Computes the distance between two arrays, according to a learned metric.

    Args:
        components (numpy.ndarray): The linear transformation deduced from the
            learned Mahalanobis metric.
        u (numpy.ndarray): The first point involved in the distance computation.
        v (numpy.ndarray): The second point involved in the distance computation.
        squared (bool, optional): If True, the function will return the squared
            metric between u and v, which is faster to compute. Defaults to False.

    Returns:
        float: The distance between u and v according to the distance metric.
    """
    transformed_diff = (u - v).dot(components)
    dist = np.dot(transformed_diff, transformed_diff.T)
    if not squared:
        dist = np.sqrt(dist)
    return dist
