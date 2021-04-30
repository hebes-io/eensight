# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np 
import pandas as pd 
import scipy.stats as stats

from sklearn.base import BaseEstimator, TransformerMixin



def fit_pdf(x, data, distribution=stats.norm):
    # fit dist to data
    params = distribution.fit(data)

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Calculate fitted PDF and error with fit in distribution
    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
    return params, pdf


def linear_impute(X, window=6):
    dt = X.index.to_series().diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()
    limit = int(window * pd.Timedelta('1H') / time_step)
    return X.interpolate(method='slinear', limit_area='inside', 
                         limit_direction='both', axis=0, limit=limit)


def maybe_reshape_2d(arr):
    # Reshape output so it's always 2-d and long
    if arr.ndim < 2:
        arr = arr.reshape(-1, 1)
    return arr



class DateFeatureTransformer(TransformerMixin, BaseEstimator):
    """
    Parameters
    ----------
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    remainder : str, :type : {'drop', 'passthrough'}, default='drop'
        By specifying ``remainder='passthrough'``, all remaining columns will be 
        automatically passed through. This subset of columns is concatenated with 
        the output of the transformer.
    """

    def __init__(self, copy_X=True, remainder='drop'):
        if remainder not in ('passthrough', 'drop'):
            raise ValueError('Parameter "remainder" should be "passthrough" or "drop"')
        self.copy_X = copy_X 
        self.remainder = remainder
    
    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('This function expects pd.DataFrame as an input')

        if self.copy_X:
            X = X.copy()

        index_dtype = X.index.dtype
        if isinstance(index_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            index_dtype = np.datetime64
        if not np.issubdtype(index_dtype, np.datetime64):
            X.index = pd.to_datetime(X.index, infer_datetime_format=True)

        attr = ['month', 'week', 'dayofweek']
        time_step = X.index.to_series().diff().min()
        if time_step < pd.Timedelta(days=1): 
            attr = attr + ['hour', 'minute']
    
        week = (X.index.isocalendar().week.astype(X.index.day.dtype) 
                if hasattr(X.index, 'isocalendar') 
                else X.index.week)
        
        if self.remainder == 'passthrough':
            for n in attr: 
                X.insert(len(X.columns), n, getattr(X.index, n) if n != 'week' else week)
        else:
            X = pd.DataFrame.from_dict(
                {n: (getattr(X.index, n) if n != 'week' else week) for n in attr}
            )
        return X 
