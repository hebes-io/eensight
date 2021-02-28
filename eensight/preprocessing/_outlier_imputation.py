# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def linear_impute(X: pd.Series, window=2) -> pd.Series:
    dt = X.index.to_series().diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()
    limit = int(window * pd.Timedelta('1H') / time_step)

    return X.interpolate(method='slinear', limit_area='inside', 
                         limit_direction='both', axis=0, limit=limit)


def iterative_impute(X: pd.DataFrame, column: str, other_features=None) -> pd.Series:
    X = X.copy()
    if other_features is None:
        other_features = []
    elif not isinstance(other_features, list):
        other_features = [other_features]
    
    X['Date'] = X.index.date
    X['Time'] = X.index.time
    X['Hour'] = X.index.hour
    X['DayOfWeek'] = X.index.dayofweek

    X_daily = X.pivot(index='Date', columns='Time', values=column)
    X_daily.index = X_daily.index.map(pd.to_datetime)
    features = [column, 'DayOfWeek',  'Hour']
    features.extend(other_features)
    imputer = IterativeImputer(estimator=DecisionTreeRegressor()).fit(X[features])
    return pd.Series(data=imputer.transform(X[features])[:, 0], index=X.index)

