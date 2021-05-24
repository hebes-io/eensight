# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np 
import pandas as pd 

from sklearn.utils import check_array


def validate_input_data(X, extra_regressors, missing, fitting=True):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('Input values are expected as pandas DataFrames (ndim=2).') 
    
    for name in extra_regressors:
        if name not in X:
            raise ValueError(f'Regressor {name} missing from dataframe')

    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    categorical_extra = [col for col in categorical_cols if col in extra_regressors]

    numeric_cols = X.select_dtypes(include='number').columns
    numeric_extra = [col for col in numeric_cols if col in extra_regressors]
    
    if fitting:
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if (len(categorical_cols) > 0) and X[categorical_cols].isnull().values.any():
            X[categorical_cols] = X[categorical_cols].fillna(value='_novalue_')
        
        if (len(numeric_cols) > 0) and X[numeric_cols].isnull().values.any():  
            if missing == 'impute':  
                X[numeric_cols] = X[numeric_cols].fillna(value=X[numeric_cols].median())
            elif missing == 'drop':
                X = X.dropna(axis=0, how='any', subset=numeric_cols)
            else:
                raise ValueError('Found missing values in input data')
        return X, categorical_extra, numeric_extra
    
    else:
        X = X.replace([np.inf, -np.inf], np.nan)

        if (len(categorical_cols) > 0) and X[categorical_cols].isnull().values.any():
            X[categorical_cols] = X[categorical_cols].fillna(value='_novalue_')
        
        if (len(numeric_cols) > 0) and X[numeric_cols].isnull().values.any():  
            X = X.dropna(axis=0, how='any', subset=numeric_cols) 
        return X 


def validate_target_data(y, index=None):
    if not isinstance(y, pd.DataFrame):
        raise ValueError('Target values are expected as pandas DataFrame (ndim=2).')
    if y.shape[1] > 1:
        raise ValueError('This estimator expects a univariate target')
    
    if index is not None:
        y = y[y.index.isin(index)]
    
    y = pd.DataFrame(data=check_array(y), index=y.index, columns=y.columns)
    return y





