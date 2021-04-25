# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import click
import mlflow
import mlflow.sklearn

import numpy as np 
import pandas as pd 
from pathlib import Path
from joblib import Parallel, delayed

from eensight.definitions import DATA_DIR
from eensight.preprocessing import validate_data, check_column_values_not_null
from eensight.preprocessing import global_filter, global_outlier_detect, local_outlier_detect
from eensight.prediction import seasonal_predict


def linear_impute(X, window=6):
    dt = X.index.to_series().diff()
    time_step = dt.iloc[dt.values.nonzero()[0]].min()
    limit = int(window * pd.Timedelta('1H') / time_step)
    return X.interpolate(method='slinear', limit_area='inside', 
                         limit_direction='both', axis=0, limit=limit)


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


def _single_pipeline(data, target_name, run_id, rebind):
    col_name = rebind.get(target_name) or target_name
    date_col_name = rebind.get('timestamp') or 'timestamp'
    data = validate_data(data, col_name, date_col_name=date_col_name) 
    data[target_name] = global_filter(
                            data[target_name], 
                            no_change_window=4,
                            allow_zero=False if target_name=='consumption' else True, 
                            allow_negative=False if target_name=='consumption' else True
    )
    pred, model = seasonal_predict(data, target_name=target_name)
    
    path = os.path.join(DATA_DIR, '06_models',  run_id)
    if not os.path.exists(path):
        os.makedirs(path)
    mlflow.sklearn.log_model(model, path)

    residuals = data[target_name] - pred[f'{target_name}_pred']
    outliers_global = global_outlier_detect(residuals)
    outliers_local = local_outlier_detect(residuals)
    outliers = np.logical_and(outliers_global, outliers_local)
    data[target_name] = data[target_name].mask(outliers, other=np.nan)
    
    if target_name == 'consumption':
        for _, group in data.groupby([lambda x: x.year, lambda x: x.month]):
            check = check_column_values_not_null(group, 'consumption', mostly=0.9)
            if not check.success:
                raise ValueError('Consumption data is not enough for baseline model development')

        data['consumption_imputed'] = False
        data['consumption_imputed'] = (
                data['consumption_imputed'].mask(data['consumption'].isna(), other=True)
        )
        data['consumption'] = (
                data['consumption'].mask(data['consumption_imputed'], other=pred['consumption_pred'])
        ) 
    return data 

    

@click.command(help='Performs the data preprocessing stage')
@click.option("--consumption", help='The path to the consumption data file', 
                               required=True, 
                               type=str)
@click.option("--temperature", help='The path to the temperature data file',
                               required=True, 
                               type=str)
@click.option("--consumption_sep", help='Delimiter to use for the energy consumption data', 
                                   default=',', 
                                   type=str, 
                                   show_default=True)
@click.option("--temperature_sep", help='Delimiter to use for the outdoor temperature data',
                                   default=',', 
                                   type=str, 
                                   show_default=True)
@click.option("--rebind", help='Map a new name onto a required one',
                          default=None, 
                          multiple=True, 
                          type=(str, str))
def preprocess_data(consumption, temperature, consumption_sep=',', 
                                              temperature_sep=',', 
                                              rebind=None):
    """
    Parameters
    ----------
    consumption : str or path object for the energy consumption data
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file.
    temperature : str or path object for the outdoor temperature data
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file.
    consumption_sep : str, default ','
        Delimiter to use for the energy consumption data.
    temperature_sep : str, default ','
        Delimiter to use for the outdoor temperature data.
    rebind : tuple of tuples, default None 
        Each nested tuple allows us to map a new name onto a required one
    """ 
    with mlflow.start_run() as mlrun:
        rebind = dict(rebind)
        consumption = dataframe_loader(consumption, sep=consumption_sep)
        temperature = dataframe_loader(temperature, sep=temperature_sep)

        consumption, temperature = (Parallel(n_jobs=2)(
                delayed(_single_pipeline)(data, target_name, mlrun.info.run_id, rebind) 
                                    for data, target_name in 
                                    zip([consumption, temperature], ['consumption', 'temperature'])
                )
        )
        data = pd.merge_asof(consumption, temperature, left_index=True, right_index=True,
                                direction='nearest', tolerance=pd.Timedelta('1H'))
        
        data[rebind.get('temperature') or 'temperature'] = linear_impute(
                    data[rebind.get('temperature') or 'temperature']
        )
        return data 
    


if __name__ == "__main__":
    preprocess_data()