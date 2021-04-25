# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math 
import stumpy

import numpy as np 
import pandas as pd 

from metric_learn import MMC
from sklearn.metrics import f1_score
from datetime import date, datetime, timedelta
from sklearn.utils import check_array
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import BaseEstimator, TransformerMixin



class MMCFeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, month_col=None, day_of_week_col=None):
        self.month_col = month_col
        self.day_of_week_col = day_of_week_col 

    def fit(self, X, y=None):
        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features) 
            Training data.
        
        Returns
        -------
        X_transformed : pd.DataFrame
        """
        index = X.index
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.to_datetime(index)
        
        if (self.month_col is None) or (self.day_of_week_col is None):
            features = pd.DataFrame.from_dict({
                            'month': index.month,
                            'dayofweek': index.dayofweek,
                        })
            features = pd.DataFrame(
                    data=np.concatenate((pd.get_dummies(features['month']).values, 
                                         pd.get_dummies(features['dayofweek']).values), axis=1),
                    index=index.date
            ) 
        else:
            # Input validation
            X = check_array(X)
            daily_data = pd.DataFrame(data=X, index=index).resample('1D').first()
            features = np.concatenate((pd.get_dummies(daily_data[self.month_col]), 
                                       pd.get_dummies(daily_data[self.day_of_week_col])), axis=1)
        return features






def get_days_to_ignore(data, start_time, end_time, threshold=0.3):
    def _ignore(x):
        pct_imputed = x['consumption_imputed'].sum() / x['consumption_imputed'].count()
        return pct_imputed > threshold
    
    should_ignore = (data.between_time(start_time, end_time, include_end=False)
                .groupby(lambda x: x.date)
                .apply(_ignore)
    )
    return should_ignore[should_ignore].index


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


def maximum_mean_discrepancy(data):
    Kyy = euclidean_distances(data)
    ny = data.shape[0]
    data_term = (np.sum(Kyy) - np.trace(Kyy)) / (ny * (ny-1))
    
    def calculate_mmd(prototypes):
        nx = prototypes.shape[0]
        Kxx = euclidean_distances(prototypes)
        Kxy = euclidean_distances(prototypes, data)

        t1 = (1./(nx*(nx-1))) * (np.sum(Kxx) - np.trace(Kxx))
        t2 = (2./(nx*ny)) * np.sum(Kxy)
        mmd2 = (t1 - t2 + data_term)
        return mmd2 / data_term
    return calculate_mmd


def find_prototypes(data, mp, start_time, col_name='consumption', 
                                          window=24, 
                                          max_iter=10, 
                                          ignored_index=None, 
                                          early_stopping=True,
                                          early_stopping_val=0.1):
    time_step = data.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)
    m = window * steps_per_hour
    end_time = (datetime.combine(date(2000,1,1), start_time) + timedelta(hours=window)).time()
    
    X = data[col_name]
    if ('date' not in data.columns) or ('time' not in data.columns):
        data = data.copy()
        data['date'] = data.index.date
        data['time'] = data.index.time

    if start_time != end_time:
        data_ = (data.between_time(start_time, end_time, include_end=False)  
                    .pivot(index='date', columns='time', values=col_name))
    else:
        data_ = (data.pivot(index='date', columns='time', values=col_name))
    
    mp_daily = mp[mp.index.time == start_time]
    
    if ignored_index is not None:
        mp_daily = mp_daily[~np.isin(mp_daily.index.date, ignored_index)]
    
    min_idx = int(mp_daily.iloc[mp_daily['nnd'].argmin()]['idx'])
    candidate = X[min_idx : min_idx+m]
    patterns = [min_idx]

    distance = pd.Series(data=stumpy.core.mass(candidate, X), index=X.index[:-m+1])
    distance = distance[distance.index.time == start_time]
    distance_from_prototypes = distance.to_frame(0)
    distance = distance.to_frame('nnd')
    
    dist_mp = None
    candidates = np.array(candidate).T.reshape(1,-1)

    if early_stopping:
        stopping_metric = []
        calculate_mmd = maximum_mean_discrepancy(data_)
    
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
        candidate = X[min_idx : min_idx+m]
        patterns.append(min_idx)
        
        distance = pd.Series(data=stumpy.core.mass(candidate, X), index=X.index[:-m+1])
        distance = distance[distance.index.time == start_time]
        distance_from_prototypes = pd.concat(
                        [distance_from_prototypes, distance.to_frame(i)], 
                        axis=1
        )
        distance = distance.to_frame('nnd')
        candidates = np.concatenate((np.array(candidate).T.reshape(1,-1), candidates))
        
        if early_stopping:
            stopping_metric.append(calculate_mmd(candidates))
            stopping_metric_ = pd.Series(stopping_metric)
            res = stopping_metric_[((stopping_metric_.abs() < early_stopping_val).rolling(2).sum()==2)]
            if len(res) > 0:
                break

    return patterns, distance_from_prototypes, stopping_metric


def create_mmc_pairs(distances, pairs_per_prototype=500):
    daily_index = distances.index.map(lambda x: x.date)
    n_pairs = int(pairs_per_prototype * distances.shape[1])
    
    positive_pairs = np.ones((n_pairs, 3), dtype=np.int32)
    negative_pairs = (-1) * np.ones((n_pairs, 3), dtype=np.int32) 

    for i, prototype in enumerate(distances.columns): 
        threshold_low = distances[prototype].quantile(0.1)
        threshold_high = distances[prototype].quantile(0.8)
        
        for j in range(pairs_per_prototype*i, pairs_per_prototype*i+pairs_per_prototype):
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



def learn_distance_metric(features, pairs):
    X_train, X_test, y_train, y_test = train_test_split(pairs[:, :2], pairs[:, -1], 
                shuffle=True, stratify=pairs[:, -1], test_size=0.5
    )

    mmc = MMC(preprocessor=features.values.astype(float))
    mmc = mmc.fit(X_train, y_train)
    score = f1_score(y_test, mmc.predict(X_test), average='weighted')
    return score, mmc
    
    