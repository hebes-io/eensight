# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro

import logging

from typing import Any, Dict, Union
from omegaconf.dictconfig import DictConfig

from eensight.io._base import AbstractDataSet
from eensight.io._memory_data_set import MemoryDataSet 



class CachedDataSet(AbstractDataSet):
    """``CachedDataSet`` is a dataset wrapper which caches in memory the data saved,
    so that the user avoids io operations with slow storage media.
    """

    def __init__(self, dataset: Union[AbstractDataSet, DictConfig], copy_mode: str = 'assign'):
        """Creates a new instance of ``CachedDataSet`` pointing to the provided Python object.
            
            Parameters
            ----------
            dataset: An AbstractDataSet object or an omegaconf.dictconfig.DictConfig to cache.
            copy_mode: The copy mode used to copy the data. Possible values are: 
                "deepcopy", "copy" and "assign". 
            
            Raise
            _____
            ValueError: If the provided dataset is not a valid DictConfig representation 
                of a dataset or an actual dataset.
        """
        if isinstance(dataset, AbstractDataSet):
            self._dataset = dataset
        elif isinstance(dataset, DictConfig):
            self._dataset = self._from_config(dataset)
        else:
            raise ValueError(
                "The argument type of `dataset` should be either a omegaconf.dictconfig.DictConfig"
                " or the actual dataset object."
            )
        self._cache = MemoryDataSet(copy_mode=copy_mode)

    
    def _describe(self) -> Dict[str, Any]:
        return {
            "dataset": self._dataset._describe(),  # pylint: disable=protected-access
            "cache": self._cache._describe(),  # pylint: disable=protected-access
        }


    def _load(self):
        data = self._cache.load() if self._cache.exists() else self._dataset.load()

        if not self._cache.exists():
            self._cache.save(data)

        return data


    def _save(self, data: Any) -> None:
        self._dataset.save(data)
        self._cache.save(data)


    def _exists(self) -> bool:
        return self._cache.exists() or self._dataset.exists()


    def _release(self) -> None:
        self._cache.release()
        self._dataset.release()


    def __getstate__(self): # returns object to be pickled
        logging.getLogger(__name__).warning("%s: clearing cache to pickle.", str(self))
        self._cache.release()
        return self.__dict__