# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math 
import stumpy

import numpy as np 
import pandas as pd 

from collections import defaultdict
from adtk.detector import InterQuartileRangeAD 
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler
from datetime import date, time, datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity




def get_matrix_profile(X, window=24):
    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)

    m = window * steps_per_hour
    mp = stumpy.stump(X, m)

    profile = pd.DataFrame(data=mp[:, :2], index=X.index[:-m+1], columns=['nnd', 'nnidx'])
    profile['idx']   = profile.reset_index().index.astype(np.int32)
    profile['nnd']   = profile['nnd'].astype(np.float)
    profile['nnidx'] = profile['nnidx'].astype(np.int32)
    return profile # matrix profile


def find_discord_days(profile, c=3):
    profile = profile[profile.index.time == time(0, 0)]
    ad = InterQuartileRangeAD(c=c)
    return ad.fit_detect(profile['nnd']) #discords


def find_patterns(X, profile, start_time, window=8, ignored_index=None, n_iter=10, threshold=0.15):
    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)

    m = window * steps_per_hour
    profile = profile[profile.index.time == start_time]
    min_idx = int(profile.iloc[profile['nnd'].argmin()]['idx'])
    reference = X[min_idx : min_idx+m]

    patterns = [min_idx]
    diss_mp = None
    stop_metric = []
    
    for i in range(n_iter):
        jmp = pd.Series(data=stumpy.core.mass(reference, X), index=X.index[:-m+1])
        jmp = jmp[jmp.index.time == start_time].to_frame('nnd')

        if diss_mp is None:
            diss_mp = jmp
        else:
            diss_mp = diss_mp.mask(jmp < diss_mp, jmp)    

        rel_profile = (profile['nnd'] / diss_mp['nnd']).to_frame()
        rel_profile['idx'] = profile['idx']
        if ignored_index is not None:
            rel_profile = rel_profile[~np.isin(rel_profile.index, ignored_index)]
        rel_profile['nnd'] = rel_profile['nnd'].clip(upper=1)

        min_idx = int(rel_profile.iloc[rel_profile['nnd'].argmin()]['idx'])
        reference = X[min_idx : min_idx+m]
        patterns.append(min_idx)
        stop_metric.append(rel_profile['nnd'].quantile(0.05))
        
    x_d = range(len(stop_metric))
    curve = UnivariateSpline(x_d, np.array(stop_metric))(x_d)
    n_patterns = 2 + np.where(abs(1 - (curve[1:] / curve[:-1]))<threshold)[0][0]
    return curve, patterns[:n_patterns] #reference patterns 


def find_similarities(X, patterns, start_time, window=8, ignored_index=None):
    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)
    m = window * steps_per_hour
    end_time = (datetime.combine(date(1970,1,1), start_time) + timedelta(hours=window)).time()

    similarity_1 = lambda a, b: 1 - (np.var(b-a) / np.var(a))
    similarity_2 = lambda a, b: cosine_similarity(np.atleast_2d(a), np.atleast_2d(b)).item()

    results = []
    for i, idx in enumerate(patterns):
        reference = X[idx: idx+m]
        res = defaultdict(list)

        for name, group in X.groupby(lambda x: x.date):
            r = similarity_1(
                        reference.values,
                        group.between_time(start_time, end_time, include_end=False).values
                )
            res[name].append(r)      

            r = similarity_2(
                        reference.values,
                        group.between_time(start_time, end_time, include_end=False).values
            )
            res[name].append(r)

        results.append(pd.DataFrame.from_dict(res, orient='index', 
                                                   columns=[f'snr_{i}', f'cos_{i}'])
        )      

    similarities = pd.concat(results, axis=1)
    if ignored_index is not None:
        similarities = similarities[~np.isin(similarities.index, ignored_index.date)]
    
    snr_columns = [col for col in similarities.columns if 'snr' in col]
    snr = similarities[snr_columns]
    similarities.loc[:, snr_columns] = (MinMaxScaler(feature_range=(0, 1))
                                         .fit_transform(snr.values.flatten()[:, None])
                                         .reshape(snr.shape))

    cos_columns = [col for col in similarities.columns if 'cos' in col]
    cos = similarities[cos_columns]
    similarities.loc[:, cos_columns] = (MinMaxScaler(feature_range=(0, 1))
                                         .fit_transform(cos.values.flatten()[:, None])
                                         .reshape(cos.shape))
    
    results = np.zeros((similarities.shape[0], len(patterns)))
    for i in range(len(patterns)):
        results[:, i] = (similarities[f'snr_{i}']**2 + similarities[f'cos_{i}']**2).map(np.sqrt).values

    return pd.DataFrame(results, index=similarities.index)


def assign_clusters(similarities, threshold=7, exclude_cluster=None):
    if exclude_cluster is not None:
        if not isinstance(exclude_cluster, list):
            exclude_cluster = [exclude_cluster]
        similarities = similarities[[col for col in similarities.columns if col not in exclude_cluster]]
        similarities = similarities.rename(columns={col: i for i, col in enumerate(similarities.columns)})
    
    clusters = similarities.idxmax(axis=1)
    counts = clusters.value_counts()
    keep = counts[counts >= threshold].index.tolist()
    similarities = similarities[keep]
    similarities = similarities.rename(columns={col: i for i, col in enumerate(similarities.columns)})
    clusters = similarities.idxmax(axis=1)
    return similarities, clusters


def create_classification_features(data, start_time, end_time, ignored_index=None):
    if ignored_index is not None:
        data = data[~np.isin(data.index, ignored_index)]
    
    X = (
            data.between_time(start_time, end_time, include_end=False)[['Temperature']]
                .groupby(lambda x: x.date)
                .max()
                .rename(columns={'Temperature': 'Temperature_max'})
    )
    
    X = (
            data.between_time(start_time, end_time, include_end=False)[['Temperature']]
                .groupby(lambda x: x.date)
                .min()
                .rename(columns={'Temperature': 'Temperature_min'})
                .join(X)
    )
    
    X.index = X.index.map(pd.to_datetime)
    X['DayOfWeek'] = X.index.dayofweek
    X['DayOfMonth'] = X.index.day
    X['MonthOfYear'] = X.index.month
    return X