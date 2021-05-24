# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro


import copy 
from typing import Any, Dict

from eensight.io._base import AbstractDataSet
from eensight.io._exceptions import DataSetError 


_EMPTY = object()


def _copy_with_mode(data: Any, copy_mode: str) -> Any:
    """Returns the copied data using the copy mode specified.
    If no copy mode is provided, it raises a DataSetError
    
    Parameters
    ----------
    data: The data to copy.
    copy_mode: The copy mode to use, one of "deepcopy", "copy" and "assign".

    Return
    ------
    The data copied according to the specified copy mode.

    Raise
    _____
    DataSetError: If copy_mode is not specified or is not valid
        (i.e: not one of deepcopy, copy, assign)    
    """
    if copy_mode == "deepcopy":
        copied_data = copy.deepcopy(data)
    elif copy_mode == "copy":
        copied_data = data.copy()
    elif copy_mode == "assign":
        copied_data = data
    else:
        raise DataSetError(
            "Invalid copy mode: {}. Possible values are: deepcopy, copy, assign.".format(
                copy_mode
            )
        )

    return copied_data


class MemoryDataSet(AbstractDataSet):
    """``MemoryDataSet`` loads and saves data from/to an in-memory
    Python object.
    """

    def __init__(self, data: Any = _EMPTY, copy_mode: str = 'assign'):
        """Creates a new instance of ``MemoryDataSet`` pointing to the
        provided Python object.
        
        Parameters
        ----------
        data: Python object containing the data.
        copy_mode: The copy mode used to copy the data. Possible
            values are: "deepcopy", "copy" and "assign".
        """
        self._data = _EMPTY
        self._copy_mode = copy_mode
        if data is not _EMPTY:
            self._save(data)


    def _save(self, data: Any):
        copy_mode = self._copy_mode
        self._data = _copy_with_mode(data, copy_mode=copy_mode)

    
    def _load(self) -> Any:
        if self._data is _EMPTY:
            raise DataSetError("Data for MemoryDataSet has not been saved yet.")

        copy_mode = self._copy_mode
        data = _copy_with_mode(self._data, copy_mode=copy_mode)
        return data
    
    
    def _exists(self) -> bool:
        return self._data is not _EMPTY


    def _release(self) -> None:
        self._data = _EMPTY


    def _describe(self) -> Dict[str, Any]:
        if self._data is not _EMPTY:
            return dict(data=f"<{type(self._data).__name__}>")
        # the string representation of datasets leaves out __init__
        # arguments that are empty/None, equivalent here is _EMPTY
        return dict(data=None)  # pragma: no cover


