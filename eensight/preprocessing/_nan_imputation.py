# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import pandas as pd

from typing import Union

from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning

from eensight.utils import as_list

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def linear_impute(X: pd.Series, window=6, copy=True) -> pd.Series:
    if copy:
        X = X.copy()

    dt = X.index.to_series().diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()
    limit = int(window * pd.Timedelta('1H') / time_step)

    return X.interpolate(method='slinear', limit_area='inside', 
                         limit_direction='both', axis=0, limit=limit)


def iterative_impute(X: Union[pd.Series, pd.DataFrame], 
                     target_name: str, 
                     other_features=None) -> pd.Series:
    
    if isinstance(X, pd.Series):
        X = X.to_frame(target_name)
    else:
        X = X.copy()
    
    other_features = as_list(other_features)
    
    X['month'] = X.index.month
    X['dayofweek'] = X.index.dayofweek
    X['hour'] = X.index.hour

    features = [target_name, 'month',  'dayofweek', 'hour']
    if other_features is not None:
        features.extend(other_features)
    
    imputer = IterativeImputer(estimator=DecisionTreeRegressor()).fit(X[features])
    return pd.Series(data=imputer.transform(X[features])[:, 0], index=X.index)

