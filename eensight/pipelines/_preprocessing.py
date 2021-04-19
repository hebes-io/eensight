# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd 
import dagster_pandas as dagster_pd

from dagster import pipeline, solid
from dagster import (Dict, Field, InputDefinition, Noneable, 
                     Optional, Output, OutputDefinition)


from eensight.preprocessing import (check_column_exists, check_column_type_datetime,
                                    check_column_values_dateutil_parseable,
                                    check_column_values_unique, check_column_values_increasing)
from eensight.preprocessing import remove_dublicate_dates, expand_to_all_dates
from eensight.preprocessing import seasonal_predict




################### Solids #################################################################
############################################################################################

@solid(
    input_defs=[
        InputDefinition(name='data', dagster_type=dagster_pd.DataFrame, description='Input data'),
        InputDefinition(name='col_name', dagster_type=str,  
                            description='The name of the column to process'),
        InputDefinition(name='date_col_name', dagster_type=Optional[str], default_value=None, 
                            description='The name of the dates column if it is different than `timestamp`'),
    ],
    output_defs=[
        OutputDefinition(name='validated_data', dagster_type=dagster_pd.DataFrame),
    ],
)
def validate_data(_, data, col_name, date_col_name):
    date_col_name = date_col_name or 'timestamp'
    
    if not (check_column_exists(data, date_col_name).success and 
            check_column_exists(data, col_name).success
    ):
        raise ValueError('Input dataset is misspecified') 

    if (not check_column_type_datetime(data, date_col_name).success and 
        check_column_values_dateutil_parseable(data, date_col_name).success
    ):
        data[date_col_name] = data[date_col_name].map(pd.to_datetime)

    if not check_column_type_datetime(data, date_col_name).success:
        raise ValueError(f'Column `{date_col_name}` must be in datetime format')

    if not check_column_values_unique(data, date_col_name).success:
        data = remove_dublicate_dates(data, date_col_name)

    if not check_column_values_increasing(data, date_col_name).success:
        data = data.sort_values(by=[date_col_name])

    data = expand_to_all_dates(data, date_col_name)
    data = data[[date_col_name, col_name]].set_index(date_col_name)
    yield Output(data, output_name='validated_data') 



@solid(
    input_defs=[
        InputDefinition(name='validated_data', dagster_type=dagster_pd.DataFrame, description='Input data'),
        InputDefinition(name='col_name', dagster_type=str,  
                            description='The name of the column to process'),
        InputDefinition(name='trend', dagster_type=Optional[str], default_value='c', 
                            description='The trend to include in the model'),
    ],
    output_defs=[
        OutputDefinition(name='seasonal_prediction', dagster_type=dagster_pd.DataFrame),
    ],
)
def apply_seasonal_predict(_, validated_data, col_name, trend):
    model, pred = seasonal_predict(validated_data, target_name=col_name, trend=trend) 
    yield Output(pred, output_name='seasonal_prediction')




################### Pipelines ##############################################################
############################################################################################


@pipeline
def preprocessing_pipeline():
    validated_data = validate_data()
    apply_seasonal_predict(validated_data)
