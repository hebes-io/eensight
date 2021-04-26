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
from eensight.definitions import DataStage
from eensight.prediction import seasonal_predict
from eensight.preprocessing import validate_data, check_column_values_not_null
from eensight.preprocessing import global_filter, global_outlier_detect, local_outlier_detect



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


def dataframe_writer(df, file_name, data_stage, run_id, sep=','):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f'Expected input type was `pandas DataFrame` but received {type(df)}')
    
    path = os.path.join(data_stage.value, run_id)
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
    

def _single_step(file_path, sep, target_name, run_id, rebind):
    col_name = rebind.get(target_name) or target_name
    date_col_name = rebind.get('timestamp') or 'timestamp'
    
    with mlflow.start_run(tags={'parent': run_id, 'target': target_name}):
        data = dataframe_loader(file_path, sep=sep)
        data = validate_data(data, col_name, date_col_name=date_col_name) 
        data[col_name] = global_filter(
                                data[col_name], 
                                no_change_window=4,
                                allow_zero=False if target_name=='consumption' else True, 
                                allow_negative=False if target_name=='consumption' else True
        )
        pred, model = seasonal_predict(data, target_name=col_name)
        mlflow.sklearn.log_model(model, artifact_path='seasonal-prediction-model')

        residuals = data[col_name] - pred[f'{col_name}_pred']
        outliers_global = global_outlier_detect(residuals)
        outliers_local = local_outlier_detect(residuals)
        outliers = np.logical_and(outliers_global, outliers_local)
        data[col_name] = data[col_name].mask(outliers, other=np.nan)
        
        if target_name == 'consumption':
            for _, group in data.groupby([lambda x: x.year, lambda x: x.month]):
                check = check_column_values_not_null(group, col_name, mostly=0.9)
                if not check.success:
                    raise ValueError('Consumption data is not enough for baseline model development')

            data[f'{col_name}_imputed'] = False
            data[f'{col_name}_imputed'] = (
                    data[f'{col_name}_imputed'].mask(data[col_name].isna(), other=True)
            )
            data[col_name] = (
                    data[col_name].mask(data[f'{col_name}_imputed'], other=pred[f'{col_name}_pred'])
            ) 

        return data 

    
@click.command(help='Given a path to an energy consumption data file and a path to an outdoor '
                    'temperature data file, performs the data preprocessing stage')
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
def preprocess_data(consumption, temperature, consumption_sep=',', temperature_sep=',', rebind=None):
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
    rebind : dictionary or tuple of tuples, default None 
        Each dictionary entry or nested tuple allows us to map a new name onto a required one (
        'consumption', 'temperature', timestamp').

    """ 
    if (rebind is not None) and not isinstance(rebind, dict):
        rebind = dict(rebind)
    elif rebind is None:
        rebind = dict()
    
    with mlflow.start_run() as mlrun:
        run_id = mlrun.info.run_id
    
        consumption, temperature = (Parallel(n_jobs=2)(
                delayed(_single_step)(file_path, sep, target_name, run_id, rebind) 
                                        for file_path, sep, target_name in 
                                        zip([consumption,     temperature], 
                                            [consumption_sep, temperature_sep],
                                            ['consumption',   'temperature'])
                )
        )
        data = pd.merge_asof(consumption, temperature, left_index=True, right_index=True,
                                direction='nearest', tolerance=pd.Timedelta('1H'))
        
        temperature_col = rebind.get('temperature') or 'temperature'
        data[temperature_col] = linear_impute(data[temperature_col])

        path = dataframe_writer(data, 'merged_data.csv', DataStage.INTERMEDIATE, run_id, sep=',')
        mlflow.log_artifacts(path, "merged-data")
    
    return data 
    


if __name__ == "__main__":
    preprocess_data()