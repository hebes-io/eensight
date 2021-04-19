# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math 
import numpy as np
import pandas as pd
import scipy.stats as stats


def global_filter(X: pd.Series, no_change_window: int=3,
                                min_value: float=None, 
                                max_value: float=None, 
                                allow_zero: bool=False, 
                                allow_negative: bool=False) -> pd.Series:
    X = X.copy() 
    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)
    no_change_window = no_change_window * steps_per_hour

    changes = X.diff().abs()
    threshold = 1e-3 * changes.max()
    X = X.mask(changes.rolling(f'{no_change_window}H').sum() < threshold, np.nan)
    
    if min_value is not None:
        X.loc[X<min_value] = np.nan
    if max_value is not None:
        X.loc[X>max_value] = np.nan
    if not allow_zero:
        X.loc[X==0] = np.nan
    if not allow_negative:
        X.loc[X<0] = np.nan

    median = X.median()
    X.loc[X.abs() > 10*median] = np.nan
    return X 


def global_outlier_detect(X: pd.Series, c: float=4) -> pd.Series:
    distribution = stats.t
    params = distribution.fit(X.dropna())

    # Separate parts of parameters
    loc = params[-2]
    scale = params[-1]

    def _detect(x):
        if np.isnan(x):
            return False
        elif (x > loc + c*scale) or (x < loc - c*scale):
            return True
        else:
            return False
    return X.map(_detect)


def local_outlier_detect(X: pd.Series, min_samples: float=0.6, c: float=4) -> pd.Series:
    X = X.to_frame()
    time_step = X.index.to_series().diff().min()
    steps_per_hour = math.ceil(pd.Timedelta('1H') / time_step)
    min_samples = int(24 * steps_per_hour * min_samples)
    
    median = X.iloc[:,0].groupby(lambda x: x.date).median()
    mad = X.iloc[:,0].groupby(lambda x: x.date).mad()
    count = X.iloc[:,0].groupby(lambda x: x.date).count()

    def _detect(x):
        idx = x.name.date()
        if np.isnan(x.iloc[0]):
            return False
        elif count[idx] < min_samples:
            return True
        elif ((x.iloc[0] < median[idx] - c * mad[idx]) or 
              (x.iloc[0] >  median[idx] + c * mad[idx])
        ):
            return True
        else:
            return False
    return X.apply(_detect, axis=1)


