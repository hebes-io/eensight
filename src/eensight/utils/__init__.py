# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import scipy
from pandas.api.types import is_bool_dtype as is_bool
from pandas.api.types import is_categorical_dtype as is_category
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_integer_dtype as is_integer
from pandas.api.types import is_object_dtype as is_object
from sklearn.utils import check_array
from sklearn.utils.validation import column_or_1d


def maybe_reshape_2d(arr):
    # Reshape output so it's always 2-d and long
    if arr.ndim < 2:
        arr = arr.reshape(-1, 1)
    return arr


def as_list(val):
    """Helper function, always returns a list of the input value."""
    if isinstance(val, str):
        return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    if val is None:
        return []
    return [val]


def as_series(x):
    """Helper function to cast an iterable to a Pandas Series object."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    else:
        return pd.Series(column_or_1d(x))


def get_categorical_cols(X, int_is_categorical=True):
    """
    Returns names of categorical columns in the input DataFrame.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input values are expected as pandas DataFrames.")

    obj_cols = []
    for col in X.columns:
        # check if it is date
        if is_datetime(X[col]):
            continue
        # check if it is bool, object or category
        if is_bool(X[col]) or is_object(X[col]) or is_category(X[col]):
            obj_cols.append(col)
            continue
        # check if it is integer
        if int_is_categorical and is_integer(X[col]):
            obj_cols.append(col)
            continue
    return obj_cols


def get_datetime_data(X, col_name=None):
    if col_name is not None:
        dt_column = X[col_name]
    else:
        dt_column = X.index.to_series()

    col_dtype = dt_column.dtype
    if isinstance(col_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        col_dtype = np.datetime64
    if not np.issubdtype(col_dtype, np.datetime64):
        dt_column = pd.to_datetime(dt_column, infer_datetime_format=True)
    return dt_column


def check_X(X, exists=None, int_is_categorical=True, return_col_info=False):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input values are expected as pandas DataFrames.")

    exists = as_list(exists)
    for name in exists:
        if name not in X:
            raise ValueError(f"Regressor {name} missing from dataframe")

    categorical_cols = get_categorical_cols(X, int_is_categorical=int_is_categorical)
    numeric_cols = X.columns.difference(categorical_cols)

    if (len(categorical_cols) > 0) and X[categorical_cols].isnull().values.any():
        raise ValueError("Found NaN values in input's categorical data")
    if (len(numeric_cols) > 0) and np.any(~np.isfinite(X[numeric_cols])):
        raise ValueError("Found NaN or Inf values in input's numerical data")

    if return_col_info:
        return X, categorical_cols, numeric_cols
    return X


def check_y(y, index=None):
    if isinstance(y, pd.DataFrame) and (y.shape[1] == 1):
        target_name = y.columns[0]
    elif isinstance(y, pd.Series):
        target_name = y.name or "_target_values_"
    else:
        raise ValueError(
            "This estimator accepts target inputs as "
            "`pd.Series` or 1D `pd.DataFrame`"
        )

    if (index is not None) and not y.index.equals(index):
        raise ValueError(
            "Input data has different index than the one "
            "that was provided for comparison"
        )

    y = pd.DataFrame(
        data=check_array(y, ensure_2d=False), index=y.index, columns=[target_name]
    )
    return y


def tensor_product(a, b, reshape=True):
    """
    Compute the tensor product of two matrices a and b
    If a is (n, m_a), b is (n, m_b),
    then the result is
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise

    Parameters
    ---------
    a : array-like of shape (n, m_a)
    b : array-like of shape (n, m_b)
    reshape : bool, default True
        whether to reshape the result to be 2-dimensional ie
        (n, m_a * m_b)
        or return a 3-dimensional tensor ie
        (n, m_a, m_b)

    Returns
    -------
    dense np.ndarray of shape
        (n, m_a * m_b) if reshape = True.
    or
        (n, m_a, m_b) otherwise
    """
    if (a.ndim != 2) or (b.ndim != 2):
        raise ValueError("Inputs must be 2-dimensional")

    na, ma = a.shape
    nb, mb = b.shape

    if na != nb:
        raise ValueError("Both arguments must have the same number of samples")

    if scipy.sparse.issparse(a):
        a = a.A
    if scipy.sparse.issparse(b):
        b = b.A

    product = a[..., :, None] * b[..., None, :]

    if reshape:
        return product.reshape(na, ma * mb)

    return product


def add_constant(data, prepend=True, has_constant="skip"):
    """
    Add a column of ones to an array.

    Parameters
    ----------
    data : array_like
        A column-ordered design matrix.
    prepend : bool
        If true, the constant is in the first column.  Else the constant is
        appended (last column).
    has_constant : str {'raise', 'add', 'skip'}
        Behavior if ``data`` already has a constant. The default will return
        data without adding another constant. If 'raise', will raise an
        error if any column has a constant value. Using 'add' will add a
        column of 1s if a constant column is present.

    Returns
    -------
    array_like
        The original values with a constant (column of ones) as the first or
        last column. Returned value type depends on input type.
    """
    x = np.asanyarray(data)
    ndim = x.ndim
    if ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        raise ValueError("Only implemented for 2-dimensional arrays")

    is_nonzero_const = np.ptp(x, axis=0) == 0
    is_nonzero_const &= np.all(x != 0.0, axis=0)
    if is_nonzero_const.any():
        if has_constant == "skip":
            return x
        elif has_constant == "raise":
            if ndim == 1:
                raise ValueError("data is constant.")
            else:
                columns = np.arange(x.shape[1])
                cols = ",".join([str(c) for c in columns[is_nonzero_const]])
                raise ValueError(f"Column(s) {cols} are constant.")

    x = [np.ones(x.shape[0]), x]
    x = x if prepend else x[::-1]
    return np.column_stack(x)
