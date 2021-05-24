# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro

import os 
import pandas as pd

from copy import deepcopy
from typing import Any, Dict

from eensight.io._exceptions import DataSetError
from eensight.io._base import AbstractVersionedDataSet



class CSVDataSet(AbstractVersionedDataSet):
    """``CSVDataSet`` loads/saves data from/to a CSV file

    Parameters
    ----------
    local_dir: str
        Path to the directory of the data file. 
    file_name: str
        The name of the data file 
    load_run_id : str (default=None)
        The id of the run that created the data file. It will act as a namespace when 
        loading the data. 
    save_run_id : str (default=None)
        The current run's id. It will act as a namespace when storing the data. 
    data_stage : str (default=None)
        The data stage according to Kedro's data engineering convention
        https://kedro.readthedocs.io/en/stable/12_faq/01_faq.html#what-is-data-engineering-convention.
    ml_stage : str (default=None)
        The type of the produced data: `train` or `test`.
    load_args: Pandas options for loading CSV files.
        Here you can find all available arguments:
        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
        All defaults are preserved.
    save_args: Pandas options for saving CSV files.
        Here you can find all available arguments:
        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html
        All defaults are preserved.
    """

    DEFAULT_LOAD_ARGS = {'sep': ','} 
    DEFAULT_SAVE_ARGS = {'sep': ','} 
    

    def __init__(self, local_dir    : str,
                       file_name    : str,  
                       load_run_id  : str = None,
                       save_run_id  : str = None,
                       data_stage   : str = None,
                       ml_stage     : str = None,
                       load_args    : Dict[str, Any] = None, 
                       save_args    : Dict[str, Any] = None) -> None:
        
        super().__init__(
            local_dir=local_dir,
            file_name=file_name,
            load_run_id=load_run_id,
            save_run_id=save_run_id,
            data_stage=data_stage,
            ml_stage=ml_stage
        )
        
        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)


    def _describe(self) -> Dict[str, Any]:
        return dict(
            local_dir=self._local_dir,
            file_name=self._file_name,
            load_run_id=self._load_run_id,
            save_run_id=self._save_run_id,
            data_stage = self._data_stage,
            ml_stage=self._ml_stage,
            load_args=self._load_args,
            save_args=self._save_args
        )


    def _load(self) -> pd.DataFrame:
        load_path = self._get_load_path()

        try:
            data = pd.read_csv(load_path, **self._load_args)
        except Exception as exc:
            message = "Failed while loading data from dataset {}.\n{}".format(
                str(self), str(exc)
            )
            raise DataSetError(message) from exc
        
        return data

    
    def _save(self, data: pd.DataFrame) -> None:
        save_path = self._get_save_path()
        base_dir = save_path.parent.resolve()
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        data.to_csv(save_path, **self._save_args)
        return save_path 


    def _exists(self) -> bool:
        try:
            load_path = self._get_load_path()
        except DataSetError:
            return False
        return os.path.exists(load_path)


    def _release(self) -> None:
        super()._release()

