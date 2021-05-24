# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro

import os 
import re
import copy 
import logging
import difflib

from pathlib import Path
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from typing import Any, Dict, Optional, List, Union 

from eensight.utils import as_list
from eensight.definitions import DataStage
from eensight.io._memory_data_set import MemoryDataSet 
from eensight.io._base import AbstractDataSet, AbstractVersionedDataSet
from eensight.io._exceptions import DataSetNotFoundError, DataSetAlreadyExistsError



#######################################################################################
############## Utility functions ######################################################
#######################################################################################

def parse_catalog_configuration(catalog: DictConfig, data_dir: str):
    """Parse the catalog configuration
    
    Parameters
    __________
    catalog : An omegaconf.dictconfig.DictConfig object
        This is the catalog configuration
    data_dir : str
        The directory where the data is stored.
    
    Return
    ______
    omegaconf.dictconfig.DictConfig: {Dataset name : configuration dictionary}
    """ 
    output = {}
    local_dir = os.path.join(data_dir, catalog.site_name)
    datasets = catalog.datasets
    levels = datasets.keys()

    for level in levels:
        subset = datasets[level]
        for name, contents in subset.items():
            output[name] = {
                'local_dir'  : local_dir,
                'file_name'  : contents.file_name,
                'type'       : contents.type,
                'data_stage' : DataStage[level.upper()].value,
                'ml_stages'  : contents.stages
            }

            if 'load_args' in contents:
                output[name].update(dict(load_args=contents.load_args))
            if 'save_args' in contents:
                output[name].update(dict(save_args=contents.save_args))

    return DictConfig(output)



########################################################################################
############### _FrozenDatasets ########################################################
########################################################################################

class _FrozenDatasets:
    """Helper class to access underlying loaded datasets"""

    def __init__(self, datasets):
        self.__dict__.update(**datasets)

    # Don't allow users to add/change attributes on the fly
    def __setattr__(self, key, value):
        msg = "Operation not allowed! "
        if key in self.__dict__.keys():
            msg += "Please change datasets through configuration."
        else:
            msg += "Please use DataCatalog.add() instead."
        raise AttributeError(msg)




########################################################################################
############### DataCatalog ############################################################
########################################################################################


class DataCatalog:
    """``DataCatalog`` stores instances of ``AbstractDataSet`` implementations
    to provide ``load`` and ``save`` capabilities from anywhere in the
    program. To use a ``DataCatalog``, you need to instantiate it with
    a dictionary of data sets. Then it will act as a single point of reference
    for your calls, relaying load and save functions to the underlying data sets.

    Parameters
    ----------
    datasets: A dictionary of dataset names and dataset instances.
    feed_dict: A feed dict with data to be added in memory.
    """

    def __init__(self, datasets: Dict[str, AbstractDataSet] = None,
                       feed_dict: Dict[str, Any] = None) -> None:
        self._datasets = dict(datasets or {})
        self.datasets = _FrozenDatasets(self._datasets)
        if feed_dict:
            self.add_feed_dict(feed_dict)

        
    @classmethod
    def from_config(cls, catalog: DictConfig, 
                         data_dir: str, 
                         dataset_names: List[str] = None,
                         ml_stages: Union[str, List[str]]=None,
                         load_versions: Union[str, Dict[str, str]] = None,
                         save_version: str = None) -> "DataCatalog":
        """Create a ``DataCatalog`` instance from configuration. This is a factory method 
        used to instantiate ``DataCatalog`` with configuration parsed from configuration files.
        
        Parameters
        ----------
        catalog : An omegaconf.dictconfig.DictConfig object
            This is the catalog configuration with information for instantiating the DataSets
        data_dir : str
            The directory where the data is stored.
        dataset_names : List of str (default=None)
            The names of the datasets to instantiate. If None, all datasets in the catalog
            configuration will be instantiated.
        ml_stages : str or list of str
            The type of the processed data for which datasets should be created: `train`, `val`, `test`.
        load_versions: str or dict (default=None)
            A mapping between dataset names and versions (i.e. MLflow run IDs) to load. If only one
            value is passed, it is assumed that it applies to all datasets.
        save_version: str (default=None)
            Version string (i.e. MLflow run ID) to be used for ``save`` operations by all datasets 
            that extend the ``AbstractVersionedDataSet`` class.

        Return
        ______
        An instantiated ``DataCatalog`` containing all specified data sets, created and ready to use.
        
        Raise
        _____
        DataSetError: When the method fails to create any of the data sets from their config.
        """
        default_load_version = None 
        
        if load_versions is None:
            load_versions = {}
        elif isinstance(load_versions, dict):
            load_versions = copy.deepcopy(load_versions)
        else:
            default_load_version = load_versions
            load_versions = {}
        
        ml_stages = as_list(ml_stages) if ml_stages is not None else ['train', 'test']
        config = parse_catalog_configuration(catalog, data_dir)
        dataset_names = as_list(dataset_names) if dataset_names is not None else config.keys()
        
        datasets = {}
        for ds_name in dataset_names:
            for stage in config[ds_name].ml_stages:
                if stage in ml_stages: 
                    ds_config = OmegaConf.masked_copy(config, [ds_name])[ds_name]
                    ds_config.pop('ml_stages')
                    ds_config.ml_stage = stage
                    ds_config.load_run_id = load_versions.get(ds_name, default_load_version)
                    ds_config.save_run_id = save_version
                    datasets[f'{ds_name}_{stage}'] = AbstractDataSet.from_config(ds_config)
            
        return cls(datasets=datasets)
    

    @property
    def _logger(self):
        return logging.getLogger('data-catalog')
    

    def add(self, dataset_name: str, dataset: AbstractDataSet, replace: bool=False) -> None:
        """Adds a new ``AbstractDataSet`` object to the ``DataCatalog``.
        
        Parameters
        ----------
        dataset_name: The dataset name.
        dataset: A ``DataSet`` object to be associated with the given dataset name.
        replace: Specifies whether replacing an existing ``DataSet`` with the same name is allowed.
        
        Raise
        _____
        DataSetAlreadyExistsError: When a dataset with the same name has already been registered 
            and replace=False
        """
        if dataset_name in self._datasets:
            if replace:
                self._logger.warning("Replacing DataSet '%s'", dataset_name)
            else:
                raise DataSetAlreadyExistsError(
                    f"DataSet '{dataset_name}' has already been registered"
                )
        self._datasets[dataset_name] = dataset
        self.datasets = _FrozenDatasets(self._datasets)


    def add_all(self, datasets: Dict[str, AbstractDataSet], replace: bool=False) -> None:
        """Adds a group of new data sets to the ``DataCatalog``.
        
        Parameters
        ----------
        datasets: A dictionary of dataset names and ``DataSet`` instances.
        replace: Specifies whether replacing an existing ``DataSet`` with the same name is allowed.
        
        Raise
        _____
        DataSetAlreadyExistsError: When a dataset with the same name has already been registered 
            and replace=False.
        """
        for name, dataset in datasets.items():
            self.add(name, dataset, replace)


    def add_feed_dict(self, feed_dict: Dict[str, Any], replace: bool=False) -> None:
        """Adds instances of ``MemoryDataSet`` through feed_dict.
        
        Parameters
        ----------
        feed_dict: A dict with data to be added in memory.
        replace: Specifies whether replacing an existing ``DataSet`` with the same name is allowed.
        """
        for dataset_name in feed_dict:
            dataset = feed_dict[dataset_name]
            
            if not isinstance(dataset, MemoryDataSet):
                dataset = MemoryDataSet(data=dataset)

            self.add(dataset_name, dataset, replace)
    
    
    def _get_dataset(self, dataset_name: str) -> AbstractDataSet:
        if dataset_name not in self._datasets:
            error_msg = f"DataSet '{dataset_name}' not found in the catalog"

            matches = difflib.get_close_matches(dataset_name, self._datasets.keys())
            if matches:
                suggestions = ", ".join(matches)  # type: ignore
                error_msg += f" - did you mean one of these instead: {suggestions}"

            raise DataSetNotFoundError(error_msg)

        dataset = self._datasets[dataset_name]
        if isinstance(dataset, AbstractVersionedDataSet):
            # we only want to return a similar-looking dataset,
            # not modify the one stored in the current catalog
            dataset = dataset._copy()
        return dataset
    
    
    def load(self, dataset_name: str) -> Any:
        """Loads a registered data set.
        
        Parameters
        ----------
        dataset_name: The name of the dataset to be loaded.
        
        Return
        ______
        The loaded data as configured.
        
        Raise
        _____
        DataSetNotFoundError: When a dataset with the given name has not yet been registered.
        """

        dataset = self._get_dataset(dataset_name)

        self._logger.info(
            "Loading data from `%s` (%s)...", dataset_name, type(dataset).__name__
        )
        return dataset.load()
 

    def save(self, dataset_name: str, data: Any) -> Path:
        """Save data to a registered data set.
        
        Parameters
        ----------
        dataset_name: The name of the dataset to save to.
        data: A data object to be saved as configured in the registered data set.
        
        Raise
        _____
        DataSetNotFoundError: When a dataset with the given name has not yet been registered.
        """
        dataset = self._get_dataset(dataset_name)

        self._logger.info("Saving data to `%s` (%s)...", dataset_name, type(dataset).__name__)
        save_path = dataset.save(data)
        return save_path


    def exists(self, dataset_name: str) -> bool:
        """Checks whether registered data set exists by calling its `exists()`
        method. Raises a warning and returns False if `exists()` is not
        implemented.
        
        Parameters
        ----------
        dataset_name: The name of the dataset to be checked.
        
        Return
        ______
        Whether the dataset exists.
        """
        try:
            dataset = self._get_dataset(dataset_name)
        except DataSetNotFoundError:
            return False
        return dataset.exists()
    
    
    def release(self, dataset_name: str):
        """Release any cached data associated with a data set
        
        Parameters
        ----------
        name: The name of the dataset to release.
        
        Raise
        _____
        DataSetNotFoundError: When a dataset with the given name has not yet been registered.
        """
        dataset = self._get_dataset(dataset_name)
        dataset.release()
    
    
    def to_list(self, regex_search: Optional[str] = None) -> List[str]:
        """
        List of all ``DataSet`` names registered in the catalog.
        This can be filtered by providing an optional regular expression
        which will only return matching keys.
        
        Parameters
        ----------
        regex_search: An optional regular expression which can be provided
            to limit the data sets returned by a particular pattern.
        
        Return
        ______
        A list of ``DataSet`` names available which match the `regex_search` criteria 
            (if provided). All dataset names are returned by default.
        
        Raise
        _____
        SyntaxError: When an invalid regex filter is provided.
        """

        if regex_search is None:
            return list(self._datasets.keys())

        if not regex_search.strip():
            logging.warning("The empty string will not match any data sets")
            return []

        try:
            pattern = re.compile(regex_search, flags=re.IGNORECASE)

        except re.error as exc:
            raise SyntaxError(
                f"Invalid regular expression provided: `{regex_search}`"
            ) from exc
        return [dset_name for dset_name in self._datasets if pattern.search(dset_name)]
    
    
    def shallow_copy(self) -> "DataCatalog":
        """Returns a shallow copy of the current catalog.
        
        Return
        ______
        Copy of the current catalog.
        """
        return DataCatalog(datasets=self._datasets)
            
    
    def __eq__(self, other):
        return self._datasets == other._datasets
        