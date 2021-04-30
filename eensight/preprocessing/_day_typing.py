# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math 
import stumpy

import numpy as np 
import pandas as pd 

from metric_learn import MMC
from sklearn.metrics import f1_score
from datetime import date, datetime
from types import SimpleNamespace
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import BaseEstimator, TransformerMixin


from eensight.preprocessing.utils import DateFeatureTransformer


def get_matrix_profile(X: pd.Series, window=24) -> pd.DataFrame:
    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)

    m = window * steps_per_hour
    mp = stumpy.stump(X, m)

    profile = pd.DataFrame(data=mp[:, :2], index=X.index[:-m+1], columns=['nnd', 'nnidx'])
    profile['idx']   = profile.reset_index().index.astype(np.int32)
    profile['nnd']   = profile['nnd'].astype(float)
    profile['nnidx'] = profile['nnidx'].astype(np.int32)
    return profile # matrix profile


def get_days_to_ignore(X, start_time, end_time, threshold=0.3):
    if isinstance(X, pd.DataFrame) and (X.shape[1] > 1):
        X = X[X.filter(like='imputed').columns]
        if X.shape[1] > 1:
            raise ValueError('Input data should include only one column '
                             'with the word "imputed" in its name')
    
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    
    if not np.all(X.isin([True, False])):
        raise ValueError('Found non-boolean values in input data')

    should_ignore = (X.between_time(start_time, end_time, include_end=False)
                      .groupby(lambda x: x.date)
                      .mean() > threshold)
    
    return should_ignore[should_ignore].index


def maximum_mean_discrepancy(data):
    Kyy = euclidean_distances(data)
    ny = data.shape[0]
    data_term = (np.sum(Kyy) - np.trace(Kyy)) / (ny * (ny-1))
    
    def calculate_mmd(prototypes):
        nx = prototypes.shape[0]
        Kxx = euclidean_distances(prototypes)
        Kxy = euclidean_distances(prototypes, data)

        t1 = (np.sum(Kxx) - np.trace(Kxx)) / (nx * (nx-1))
        t2 = 2.*np.mean(Kxy)
       
        return (t1 - t2 + data_term) / data_term
    return calculate_mmd


def find_prototypes(X, mp, start_time, end_time=None, 
                                       max_iter=30, 
                                       ignored_index=None, 
                                       early_stopping=True,
                                       early_stopping_val=0.1):
    
    if not isinstance(X, pd.Series):
        raise ValueError('This function expects pd.Series as an input')

    if start_time == end_time:
        end_time = None 
    
    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)
    
    if end_time is not None:
        date_diff = (datetime.combine(date.today(), end_time) 
                    - datetime.combine(date.today(), start_time)
        )
        m = int(date_diff.total_seconds() * steps_per_hour / 3600)
    else:
        m = 24 * steps_per_hour
    
    data = X.to_frame('values')
    data['date'] = data.index.date
    data['time'] = data.index.time

    if end_time is not None:
        data_ = (data.between_time(start_time, end_time, include_end=False)  
                    .pivot(index='date', columns='time', values='values'))
    else:
        data_ = (data.pivot(index='date', columns='time', values='values'))
    
    dist_mp = None
    distance_from_prototypes = []
    mp_daily = mp[mp.index.time == start_time]
    
    if ignored_index is not None:
        mp_daily = mp_daily[~np.isin(mp_daily.index.date, ignored_index)]
    
    min_idx = int(mp_daily.iloc[mp_daily['nnd'].argmin()]['idx'])
    prototype = X[min_idx : min_idx+m]
    patterns = [min_idx]

    distance = pd.Series(data=stumpy.core.mass(prototype, X), index=X.index[:-m+1])
    distance = distance[distance.index.time == start_time]
    distance_from_prototypes.append(distance.to_frame(0))
    distance = distance.to_frame('nnd')
    
    if early_stopping:
        stopping_metric = []
        calculate_mmd = maximum_mean_discrepancy(data_)
        prototypes = np.array(prototype).T.reshape(1,-1)

    for i in range(1, max_iter):
        if ignored_index is not None:
            distance = distance[~np.isin(distance.index.date, ignored_index)]
        
        if dist_mp is None:
            dist_mp = distance
        else:
            dist_mp = dist_mp.mask(distance < dist_mp, distance) 
            
        rel_profile = (mp_daily['nnd'] / dist_mp['nnd']).to_frame()
        rel_profile['idx'] = mp_daily['idx']
        rel_profile['nnd'] = rel_profile['nnd'].clip(upper=1)
    
        threshold = rel_profile['nnd'].quantile(0.01)
        min_idx = int(rel_profile[rel_profile['nnd'] <= threshold].sample()['idx'].item())
        prototype = X[min_idx : min_idx+m]
        patterns.append(min_idx)
        
        distance = pd.Series(data=stumpy.core.mass(prototype, X), index=X.index[:-m+1])
        distance = distance[distance.index.time == start_time]
        distance_from_prototypes.append(distance.to_frame(i))
        distance = distance.to_frame('nnd')
        
        if early_stopping:
            prototypes = np.concatenate((np.array(prototype).T.reshape(1,-1), prototypes))
            stopping_metric.append(calculate_mmd(prototypes))
            stopping_metric_ = pd.Series(stopping_metric)
            res = stopping_metric_[((stopping_metric_.abs() < early_stopping_val).rolling(2).sum()==2)]
            if len(res) > 0:
                break

    return SimpleNamespace(
        prototypes=patterns,
        distance_from_prototypes=pd.concat(distance_from_prototypes, axis=1),
        stopping_metric=None if not early_stopping else stopping_metric  
    )
    

def create_mmc_pairs(distances, pairs_per_prototype=100):
    daily_index = distances.index.map(lambda x: x.date)
    n_pairs = int(pairs_per_prototype * distances.shape[1])
    
    positive_pairs = np.ones((n_pairs, 3), dtype=np.int32)
    negative_pairs = (-1) * np.ones((n_pairs, 3), dtype=np.int32) 

    for i, prototype in enumerate(distances.columns): 
        threshold_low = distances[prototype].quantile(0.1)
        threshold_high = distances[prototype].quantile(0.8)
        
        start = pairs_per_prototype*i
        end = pairs_per_prototype*i + pairs_per_prototype
        
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


class MMCFeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, month_col=None, day_of_week_col=None):
        self.month_col = month_col or 'month'
        self.day_of_week_col = day_of_week_col or 'dayofweek'

    def fit(self, X, y=None):
        # Return the transformer
        return self

    def transform(self, X):        
        if not isinstance(X.index, pd.DatetimeIndex):
            X.index = pd.to_datetime(X.index)
        
        X = pd.DataFrame(data=check_array(X), index=X.index, columns=X.columns)
        X = X.resample('1D').first()
        
        features = pd.DataFrame(0, index=X.index, columns=range(19))
        
        temp = pd.get_dummies(X[self.month_col].astype(int))
        temp.index = X.index
        for col in temp.columns:
            features[col-1] = features[col-1].mask(temp[col]==1, other=1)

        temp = pd.get_dummies(X[self.day_of_week_col].astype(int))
        temp.index = X.index
        for col in temp.columns:
            features[col+12] = features[col+12].mask(temp[col]==1, other=1)

        return pd.DataFrame(data=features, index=X.index)


def learn_distance_metric(distances, pairs_per_prototype=100, 
                                     test_size=0.5, 
                                     return_features=False,
                                     return_pairs=False):
    feature_pipeline = Pipeline([
        ('dates', DateFeatureTransformer()),
        ('features', MMCFeatureTransformer()),
    ])
    
    features = feature_pipeline.fit_transform(distances)
    pairs = create_mmc_pairs(distances, pairs_per_prototype=pairs_per_prototype)
    
    X_train, X_test, y_train, y_test = train_test_split(pairs[:, :2], pairs[:, -1], 
                shuffle=True, stratify=pairs[:, -1], test_size=test_size
    )

    mmc = MMC(preprocessor=np.array(features, dtype=np.float))
    mmc = mmc.fit(X_train, y_train)
    score = f1_score(y_test, mmc.predict(X_test), average='weighted')
    return SimpleNamespace(
        score=score,
        metric_components=mmc.components_.transpose(),
        features=None if not return_features else features,
        pairs=None if not return_pairs else pairs 
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
    
    