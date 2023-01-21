# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Union

import numpy as np
import pandas as pd
import scipy
from kedro.framework.session import KedroSession
from pandas.api.types import is_bool_dtype as is_bool
from pandas.api.types import is_categorical_dtype as is_category
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_integer_dtype as is_integer
from pandas.api.types import is_object_dtype as is_object
from sklearn.utils.validation import column_or_1d

from eensight.framework.startup import bootstrap_project
from eensight.settings import PROJECT_PATH


def as_list(val: Any) -> list:
    """Cast input as list.

    Helper function, always returns a list of the input value.
    """
    if isinstance(val, str):
        return [val]
    if hasattr(val, "__iter__"):
        return list(val)
    if val is None:
        return []
    return [val]


def as_series(x: Union[np.ndarray, pd.Series, pd.DataFrame]):
    """Cast an iterable to a Pandas Series object."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    else:
        return pd.Series(column_or_1d(x))


def get_categorical_cols(X: pd.DataFrame, int_is_categorical: bool = False) -> list:
    """Return the names of the categorical columns in the input DataFrame.

    Args:
        X (pandas.DataFrame): Input dataframe.
        int_is_categorical (bool, optional): If True, integer types are
            considered categorical. Defaults to False.

    Returns:
        list: The names of categorical columns in the input DataFrame.
    """
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


def tensor_product(a: np.ndarray, b: np.ndarray, reshape=True):
    """Compute the tensor product of two matrices.

    Args:
        a (numpy array of shape (n, m_a)): The first matrix.
        b (numpy array of shape (n, m_b)): The second matrix.
        reshape (bool, optional): Whether to reshape the result to be 2D (n, m_a * m_b)
            or return a 3D tensor (n, m_a, m_b). Defaults to True.

    Raises:
        ValueError: If input arrays are not 2-dimensional.
        ValueError: If both input arrays do not have the same number of samples.

    Returns:
        numpy.ndarray of shape (n, m_a * m_b) if `reshape = True` else of shape (n, m_a, m_b).
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


def load_catalog(store_uri: str, site_id: str, namespace: str, run_id: str = None):
    """Load the data catalog for the provided building id and namespace.

    Args:
        store_uri (str): The URI where the generated data and models should be stored.
            It is expected to be a local path. In addition, the function will use
            `{store_uri}/{site_id}/01_raw/{namespace}/features` path as default for the
            input features and `{store_uri}/{site_id}/01_raw/{namespace}/labels` for the
            input labels.
        site_id (str): The id of the site/building to load (must be one of
            the datasets already provided by `eensight`).
        namespace (str): The namespace for which to load data (`train`, `test`
            or `apply`). Not all datasets already provided by `eensight` have
            data for all namespaces.
        run_id (str, optional): The ID string for the run that contains the artifacts
            to be used. If not provided, the untracked artifacts will be used.

    Returns:
        kedro.io.DataCatalog: The data catalog.
    """

    extra_params = {
        "app": {
            "store_uri": os.path.abspath(store_uri),
            "site_id": site_id,
            "namespace": namespace,
            "run_id": run_id,
            "features": {
                "format": "csv",
                "load_args": {},
            },
            "labels": {"format": "csv", "load_args": {}},
        }
    }

    bootstrap_project(PROJECT_PATH)
    with KedroSession.create(
        package_name="eensight",
        project_path=PROJECT_PATH,
        extra_params=extra_params,
        save_on_close=False,
    ) as session:
        catalog = session.load_context().catalog
        catalog._logger.setLevel(logging.WARNING)
        return catalog
