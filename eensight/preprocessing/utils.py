# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np 
import pandas as pd 
import scipy.stats as stats



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


def as_list(val):
    """
    Helper function, always returns a list of the input value.
    """
    if isinstance(val, str):
        return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    if val is None:
        return []
    return [val]