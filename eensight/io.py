# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd 
from pathlib import Path



def dataframe_loader(file_path, sep=','):
    """
    Parameters
    ----------
    file_path : str or path object for the data
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file.
    sep : str, default ','
        Delimiter to use. If sep is None, the C engine cannot automatically detect
        the separator, but the Python parsing engine can, meaning the latter will
        be used.
    """ 
    if isinstance(file_path, str):
        file_path = os.path.normpath(file_path) 
        _, file_type = os.path.splitext(file_path)
        _, file_type = file_type.split('.')
    elif isinstance(file_path, Path):
        _, file_type = file_path.suffix.split('.')
    else:
        raise ValueError(f'File path {file_path} not understood')

    if file_type == "csv":
        return pd.read_csv(file_path, sep=sep)
    elif file_type == "parquet":
        return pd.read_parquet(file_path)
    elif file_type == "table":
        return pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f'Unsupported file_type {file_type}')


def dataframe_writer(df, file_name, namespace, sep=',', data_stage=None, data_type=None):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f'Expected input type was `pandas DataFrame` but received {type(df)}')
    
    #use namespace/data_type as a namespace
    path = ( os.path.join(data_stage.value, namespace) 
             if data_type is None 
             else os.path.join(data_stage.value, namespace, data_type)
    )
    if not os.path.exists(path):
        os.makedirs(path)
        
    _, file_type = file_name.split('.')   
    
    if file_type == 'csv':
        df.to_csv(os.path.join(path, file_name), sep=sep)
    elif file_type == 'parquet':
        df.to_parquet(os.path.join(path, file_name))
    elif file_type == 'table':
        df.to_csv(os.path.join(path, file_name), sep="\t", index=False)
    else:
        raise ValueError(f'Unsupported file_type {file_type}')
    
    return path