# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import click
import mlflow
import mlflow.sklearn

import numpy as np 
import pandas as pd 

from eensight.definitions import DataStage
from eensight.prediction import seasonal_predict
from eensight.preprocessing.utils import linear_impute
from eensight.io import dataframe_loader, dataframe_writer
from eensight.preprocessing import validate_data, check_column_values_not_null
from eensight.preprocessing import global_filter, global_outlier_detect, local_outlier_detect



def _consumption_step(file_path, sep, namespace, rebind):
    col_name = rebind.get('consumption') or 'consumption'
    date_col_name = rebind.get('timestamp') or 'timestamp'
    
    tags = {'parent': namespace, 'applied_on': 'consumption'}
    
    with mlflow.start_run(nested=True, tags=tags):
        data = dataframe_loader(file_path, sep=sep)
        data = validate_data(data, col_name, date_col_name=date_col_name) 
        data[col_name] = global_filter(data[col_name], no_change_window=4,
                                                       allow_zero=False, 
                                                       allow_negative=False
        )
        seasonal = seasonal_predict(data, target_name=col_name, return_model=True)
        mlflow.sklearn.log_model(seasonal.model, artifact_path='seasonal-prediction-model')

        outliers_global = global_outlier_detect(seasonal.resid)
        outliers_local = local_outlier_detect(seasonal.resid)
        outliers = np.logical_and(outliers_global, outliers_local)
        data[col_name] = data[col_name].mask(outliers, other=np.nan)
        
        for _, group in data.groupby([lambda x: x.year, lambda x: x.month]):
            check = check_column_values_not_null(group, col_name, mostly=0.9)
            if not check.success:
                raise ValueError('Consumption data is not enough for baseline model development')

        data[f'{col_name}_imputed'] = False
        data[f'{col_name}_imputed'] = (
                data[f'{col_name}_imputed'].mask(cond=data[col_name].isna(), 
                                                 other=True
                )
        )
        data[col_name] = (
                data[col_name].mask(cond=data[f'{col_name}_imputed'], 
                                    other=seasonal.predicted[f'{col_name}_pred']
                )
        ) 
        return data 


def _temperature_step(file_path, sep, namespace, rebind):
    col_name = rebind.get('temperature') or 'temperature'
    date_col_name = rebind.get('timestamp') or 'timestamp'
    
    tags = {'parent': namespace, 'applied_on': 'temperature'}
    
    with mlflow.start_run(nested=True, tags=tags):
        data = dataframe_loader(file_path, sep=sep)
        data = validate_data(data, col_name, date_col_name=date_col_name) 
        data[col_name] = global_filter(data[col_name], no_change_window=4,
                                                       allow_zero=True, 
                                                       allow_negative=True
        )
    
        outliers_global = global_outlier_detect(data[col_name])
        outliers_local = local_outlier_detect(data[col_name])
        outliers = np.logical_and(outliers_global, outliers_local)
        
        data[col_name] = data[col_name].mask(outliers, other=np.nan)
        data[col_name] = linear_impute(data[col_name])
        return data


def _holidays_step(file_path, sep, namespace, rebind):
    col_name = rebind.get('holiday') or 'holiday'
    date_col_name = rebind.get('timestamp') or 'timestamp'
    
    tags = {'parent': namespace, 'applied_on': 'holidays'}
    
    with mlflow.start_run(nested=True, tags=tags):
        data = dataframe_loader(file_path, sep=sep)
        data = validate_data(data, col_name, date_col_name=date_col_name) 
        return data

    

@click.command(help='Given the paths to an energy consumption data file, an outdoor '
                    'temperature data file and a holidays data file, performs the data '
                    'preprocessing stage')
@click.option("--consumption-path", help='The path to the consumption data file', 
                                    required=True, 
                                    type=str)
@click.option("--temperature-path", help='The path to the temperature data file',
                                    required=True, 
                                    type=str)
@click.option("--holidays-path",    help='The path to the public holidays data file',
                                    default=None, 
                                    type=str)
@click.option("--consumption-sep",  help='Delimiter to use for the energy consumption data', 
                                    default=',', 
                                    type=str, 
                                    show_default=True)
@click.option("--temperature-sep",  help='Delimiter to use for the outdoor temperature data',
                                    default=',', 
                                    type=str, 
                                    show_default=True)
@click.option("--holidays-sep",     help='Delimiter to use for the public holidays data',
                                    default=',', 
                                    type=str, 
                                    show_default=True)
@click.option("--run-name",         help='A name for the run. It will act as a namespace when '
                                         'storing processed data',
                                    default=None, 
                                    type=str, 
                                    show_default=True)
@click.option("--data-type",        help='The type of the produced data (if any): `train` or `test`',
                                    default='train', 
                                    type=str, 
                                    show_default=True)
@click.option("--rebind",           help='Map a new name onto one of `consumption`, `temperature`'
                                         ' `holiday` and `timestamp`',
                                    default=None, 
                                    multiple=True, 
                                    type=(str, str))
def preprocess_data(consumption_path, temperature_path, holidays_path=None,
                                                        consumption_sep=',', 
                                                        temperature_sep=',', 
                                                        holidays_sep=',', 
                                                        rebind=None,
                                                        run_name=None,
                                                        data_type=None):
    """
    Parameters
    ----------
    consumption_path : str or path object for the energy consumption data
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file.
    temperature_path : str or path object for the outdoor temperature data
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file.
    holidays_path : str or path object for the public holidays data (default=None)
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file.
    consumption_sep : str (default=',')
        Delimiter to use for the energy consumption data.
    temperature_sep : str (default=',')
        Delimiter to use for the outdoor temperature data.
    holidays_sep : str (default=',')
        Delimiter to use for the public holidays data.
    rebind : tuple of tuples (default=None) 
        Each nested tuple allows us to map a new name onto a required one (
        'consumption', 'temperature', timestamp').
    run_name : str (default=None)
        A name for the run. It will act as a namespace when storing processed data.
    data_type : str (default=None)
        The type of the produced data (if any): `train` or `test`.

    """ 
    rebind = dict() if rebind is None else dict(rebind)
    
    with mlflow.start_run(run_name=run_name) as mlrun:
        namespace = mlrun.info.run_id if run_name is None else run_name

        consumption =_consumption_step(consumption_path, consumption_sep, namespace, rebind)
        temperature = _temperature_step(temperature_path, temperature_sep, namespace, rebind)

        holidays = None
        if holidays_path is not None:
            holidays = _holidays_step(holidays_path, holidays_sep, namespace, rebind)

        data = pd.merge_asof(consumption, temperature, left_index=True, right_index=True,
                                direction='nearest', tolerance=pd.Timedelta('1H'))
        
        if holidays is not None:
            time_step = data.index.to_series().diff().min()
            data = data.join(holidays.resample(time_step).first().groupby(lambda x: x.date).pad())
            
        path = dataframe_writer(data, 'merged_data.csv', namespace, 
                                            sep=',',
                                            data_stage=DataStage.INTERMEDIATE, 
                                            data_type=data_type
        )
        
        artifact_path = 'merged-data' if data_type is None else f'merged-data-{data_type}'  
        mlflow.log_artifacts(path, artifact_path)
    return data 
    

if __name__ == "__main__":
    preprocess_data()