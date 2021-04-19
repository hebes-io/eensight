# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os 
from types import SimpleNamespace
from dagster import execute_pipeline

from eensight.pipelines import preprocessing_pipeline


def extract_results(result, return_all_steps=False):
    all_steps = result.solid_result_list
    if return_all_steps:
        return SimpleNamespace(success=result.success, 
                               value=[result.output_value() for result in all_steps])
    else:
        return SimpleNamespace(success=result.success, 
                               value=all_steps[-1].output_value())



def execute_preprocessing(file_path: str, col_name: str, date_col_name: str=None, 
                            trend: str=None, return_all_steps=False):
    
    _, file_name = os.path.split(file_path)
    _, file_type = file_name.split('.')
    
    run_config = {
        'solids': {
            'validate_data': {
                'inputs': {
                    'data': {f'{file_type}': {'path': file_path}},
                    'col_name': {'value': col_name}, 
                }
            },
            'apply_seasonal_predict': {
                'inputs': {
                    'col_name': {'value': col_name}, 
                }
            }
        }
    }
    
    if date_col_name is not None:
        run_config['solids']['validate_data']['inputs']['date_col_name'] = {'value': date_col_name}
    if trend is not None:
        run_config['solids']['apply_seasonal_predict']['inputs']['trend'] = {'value': trend}
    
    result = execute_pipeline(preprocessing_pipeline, run_config=run_config)
    return result #extract_results(result, return_all_steps=return_all_steps)