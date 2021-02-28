# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import numpy as np
import pandas as pd 
from types import SimpleNamespace


VALID_DATA_TYPES  = (pd.DataFrame, pd.Series)
VALID_INDEX_TYPES = (pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)


################### Validation checks ######################################################
############################################################################################

def check_unique_dates(index):
    if not isinstance(index, VALID_INDEX_TYPES):
        raise TypeError(
            f"{type(index)} is not supported, use one of {VALID_INDEX_TYPES} instead."
        )
    
    if index.dtype == np.int64:
        index = pd.to_datetime(index.astype(str))
    
    try:
        count = index.to_series().diff().value_counts()[pd.Timedelta('0 days 00:00:00')]
    except KeyError:
        return SimpleNamespace(success=True, meta=dict())
    else:
        return SimpleNamespace(success=False, meta=dict(count=count))


def check_no_dates_missing(index):
    if not isinstance(index, VALID_INDEX_TYPES):
        raise TypeError(
            f"{type(index)} is not supported, use one of {VALID_INDEX_TYPES} instead."
        )
    
    time_step = index.to_series().diff().min()
    full_index = pd.date_range(start=datetime.datetime.combine(
                                       index.min().date(), 
                                       datetime.time(0, 0)), 
                               end=datetime.datetime.combine(
                                    index.max().date()+datetime.timedelta(days=1), 
                                    datetime.time(0, 0)),
                               freq=time_step)[:-1]
  
    if len(full_index) > len(index):
        return SimpleNamespace(success=False, meta=dict(full_index=full_index))
    else:
        return SimpleNamespace(success=True, meta=dict())


def check_enough_data(data, column):
    result = (data[[column]].groupby([lambda s: s.year, lambda s: s.month])
                            .agg([lambda x: x.count(), lambda x: x.isna().sum()])
                            .rename({'<lambda_0>': 'count', '<lambda_1>': 'na'}, axis='columns'))[column]
    result.index = result.index.map(lambda x: datetime.date(x[0], x[1], 1))
    result = result['na']/result['count']

    for _, group in result.groupby(lambda x: x.year):
        if sum(group < 0.1) == 12:
            return SimpleNamespace(success=True, meta=dict())
    return SimpleNamespace(success=False, meta=dict(missing=result))


################### Validation actions #####################################################
############################################################################################


def remove_dublicate_dates(data, threshold=None, copy=False):
    if threshold is None:
        threshold = 0.2
    
    if not isinstance(data, VALID_DATA_TYPES):
        raise TypeError(
            f"{type(data)} is not supported, use one of {VALID_DATA_TYPES} instead."
        )
    
    name = None
    if isinstance(data, pd.DataFrame):
        name = data.columns[0]
        data = data.iloc[:, 0]

    if copy:
        data = data.copy()
    
    dupl_range = (data[data.index.duplicated(keep=False)]
                                    .groupby([lambda x: x])
                                    .agg([lambda x: x.max() - x.min(), lambda x: x.mean()])
    )
    dupl_range.columns = ['range', 'mean']
    
    data = data[~data.index.duplicated(keep='first')]
    data.loc[data.index.isin(dupl_range.index)] = np.nan
    
    fill_index = dupl_range[dupl_range['range'] < threshold * data.mean()].index
    fill_values = dupl_range.loc[fill_index]['mean'].values
    data.loc[data.index.isin(fill_index)] = fill_values
    
    if name is not None:
        return data.to_frame(name)
    else:
        return data


def add_all_dates(data, full_index):
    index_name = data.index.name
    data = pd.DataFrame(index=full_index).join(data, how='left')
    data.index.set_names(index_name, inplace=True)
    return data 