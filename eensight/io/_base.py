# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro

import os 
import abc
import copy
import logging
import importlib

from pathlib import Path
from typing import Any, Dict
from omegaconf.dictconfig import DictConfig

from ._exceptions import DataSetError



#######################################################################################
############## Utility functions ######################################################
#######################################################################################


def load_obj(obj_path: str) -> Any:
    """Extract an object from a given path.
    
    Parameters
    __________
    obj_path: Full class path to an object to be extracted, including the object name.
        
    Return
    ______
    Extracted object.
    
    Raise
    _____
    AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0)
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def parse_dataset_definition(config: DictConfig):
    """Parse and instantiate a dataset class using the configuration provided.
    
    Parameters
    __________
    config: An omegaconf.dictconfig.DictConfig object
        This is the dataset config dictionary. It must contain the `type` key
        with fully qualified class name.
        
    Raise
    _____
    DataSetError: If the function fails to parse the configuration provided.
    
    Return
    ______
    2-tuple: (Dataset class object, configuration dictionary)
    """
    if "type" not in config:
        raise DataSetError("`type` is missing from DataSet catalog configuration")

    class_obj = config.pop("type")
    if isinstance(class_obj, str):
        if len(class_obj.strip(".")) != len(class_obj): #check if starts or ends with a dot
            raise DataSetError(
                "`type` class path does not support relative "
                "paths or paths ending with a dot."
            )
        class_obj = load_obj(class_obj)
        
    if not issubclass(class_obj, AbstractDataSet):
        raise DataSetError(
            f"DataSet type `{class_obj.__module__}.{class_obj.__qualname__}` "
            f"is invalid: all data set types must extend `AbstractDataSet`."
        )

    return class_obj, config



########################################################################################
############### AbstractDataSet ########################################################
########################################################################################


class AbstractDataSet(abc.ABC):
    """``AbstractDataSet`` is the base class for all dataset implementations.
    """

    @classmethod
    def from_config(cls, config: DictConfig) -> 'AbstractDataSet':
        """Create a dataset instance using the configuration provided.
        
        Parameters
        ----------
        config: An omegaconf.dictconfig.DictConfig object.
        
        Return
        ______
        An instance of an ``AbstractDataSet`` subclass.
        
        Raise
        _____
        ValueError: When the input `config` is not an omegaconf.dictconfig.DictConfig
        DataSetError: When the function fails to create the data set from its config.
        """
        if not isinstance(config, DictConfig):
            raise ValueError('Input `config` must be an omegaconf.dictconfig.DictConfig')

        try:
            class_obj, config = parse_dataset_definition(config)
        except Exception as exc:
            raise DataSetError(
                f"An exception occurred when parsing config: {str(exc)}"
            ) from exc

        try:
            data_set = class_obj(**config)  # type: ignore
        except Exception as err:
            raise DataSetError(
                f"\n{err}.\nFailed to instantiate DataSet "
                f"of type `{class_obj.__module__}.{class_obj.__qualname__}`."
            ) from err

        return data_set
    
    
    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__file__)


    def load(self) -> Any:
        """Loads data by delegation to the provided load method.
        
        Return
        ------
        Data returned by the provided load method.
        
        Raise
        _____
        DataSetError: When underlying load method raises error.
        """

        self._logger.debug("Loading %s", str(self))

        try:
            return self._load()
        except DataSetError:
            raise
        except Exception as exc:
            # This exception handling is by design as the composed data sets
            # can throw any type of exception.
            message = "Failed while loading data from data set {}.\n{}".format(
                str(self), str(exc)
            )
            raise DataSetError(message) from exc


    def save(self, data: Any) -> Path:
        """Saves data by delegation to the provided save method.
        
        Parameters
        ----------
        data: the value to be saved by provided save method.

        Return
        ------
        The path where the data was saved.
        
        Raise
        _____
        DataSetError: when underlying save method raises error.
        """

        if data is None:
            raise DataSetError("Saving `None` to a `DataSet` is not allowed")

        self._logger.debug("Saving %s", str(self))

        try:
            save_path = self._save(data)
        except DataSetError:
            raise
        except Exception as exc:
            message = f"Failed while saving data to data set {str(self)}.\n{str(exc)}"
            raise DataSetError(message) from exc
        
        return save_path 


    def __str__(self):
        def _to_str(obj, is_root=False):
            """Returns a string representation where
            1. The root level (i.e. the DataSet.__init__ arguments) are
            formatted like DataSet(key=value).
            2. Dictionaries have the keys alphabetically sorted recursively.
            3. None values are not shown.
            """

            fmt = "{}={}" if is_root else "'{}': {}"  # 1

            if isinstance(obj, dict):
                sorted_dict = sorted(obj.items(), key=lambda pair: str(pair[0]))  # 2

                text = ", ".join(
                    fmt.format(key, _to_str(value))  # 2
                    for key, value in sorted_dict
                    if value is not None  # 3
                )
                return text if is_root else "{" + text + "}"  # 1
            # not a dictionary
            return str(obj)

        return f"{type(self).__name__}({_to_str(self._describe(), True)})"


    @abc.abstractmethod
    def _load(self) -> Any:
        raise NotImplementedError(
            "`{}` is a subclass of AbstractDataSet and"
            "it must implement the `_load` method".format(self.__class__.__name__)
        )


    @abc.abstractmethod
    def _save(self, data: Any) -> Path:
        raise NotImplementedError(
            "`{}` is a subclass of AbstractDataSet and"
            "it must implement the `_save` method".format(self.__class__.__name__)
        )

    @abc.abstractmethod
    def _describe(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "`{}` is a subclass of AbstractDataSet and"
            "it must implement the `_describe` method".format(self.__class__.__name__)
        )


    def exists(self) -> bool:
        """Checks whether a data set's output already exists by calling
        the provided _exists() method.
        
        Return
        ______
        Flag indicating whether the output already exists.
        
        Raise
        _____
        DataSetError: when underlying exists method raises error.
        """
        self._logger.debug("Checking whether target of %s exists", str(self))

        try:
            return self._exists()
        except Exception as exc:
            message = "Failed during exists check for data set {}.\n{}".format(
                str(self), str(exc)
            )
            raise DataSetError(message) from exc


    def _exists(self) -> bool:
        self._logger.warning(
            "`exists()` not implemented for `%s`. Assuming output does not exist.",
            self.__class__.__name__,
        )
        return False


    def release(self) -> None:
        """Release any cached data.
        
        Raise
        _____
        DataSetError: when underlying release method raises error.
        """
        self._logger.debug("Releasing %s", str(self))

        try:
            self._release()
        except Exception as exc:
            message = f"Failed during release for data set {str(self)}.\n{str(exc)}"
            raise DataSetError(message) from exc


    def _release(self) -> None:
        pass


    def _copy(self, **overwrite_params) -> "AbstractDataSet":
        dataset_copy = copy.deepcopy(self)
        for name, value in overwrite_params.items():
            setattr(dataset_copy, name, value)
        return dataset_copy



########################################################################################
############### AbstractVersionedDataSet ###############################################
########################################################################################


class AbstractVersionedDataSet(AbstractDataSet, abc.ABC):
    """
    ``AbstractVersionedDataSet`` is the base class for all versioned dataset implementations.

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
        The current run's id. It will act as a namespace when saving/storing the data.
    data_stage : str (default=None)
        The data stage according to Kedro's data engineering convention
        https://kedro.readthedocs.io/en/stable/12_faq/01_faq.html#what-is-data-engineering-convention.
    ml_stage : str (default=None)
        The type of the produced data: `train`, `val` or `test`.
    """

    def __init__(self, local_dir    : str,
                       file_name    : str, 
                       load_run_id  : str = None,
                       save_run_id  : str = None,
                       data_stage   : str = None,
                       ml_stage     : str = None
    ):
        self._local_dir = Path(local_dir).absolute()
        self._file_name = file_name
        self._load_run_id = load_run_id
        self._save_run_id = save_run_id
        self._data_stage = data_stage
        self._ml_stage = ml_stage
        
    
    def _get_load_path(self) -> Path:
        if self._load_run_id is None:
            path = os.path.join(self._local_dir, self._data_stage, self._ml_stage, self._file_name)
        else:
            path = os.path.join(self._local_dir, 'mlruns', self._load_run_id,
                                self._data_stage, self._ml_stage, self._file_name
            )
        return Path(path)

            
    def _get_save_path(self) -> Path:
        if self._save_run_id is None:
            path = os.path.join(self._local_dir, self._data_stage, self._ml_stage, self._file_name)
        else:
            path = os.path.join(self._local_dir, 'mlruns', self._save_run_id,
                                self._data_stage, self._ml_stage, self._file_name
            )    
        return Path(path)


