# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/quantumblacklabs/kedro

class DataSetError(Exception):
    """``DataSetError`` raised by ``AbstractDataSet`` implementations
    in case of failure of input/output methods.
    """
    pass


class DataSetNotFoundError(DataSetError):
    """``DataSetNotFoundError`` raised by ``DataCatalog`` class in case of
    trying to use a non-existing data set.
    """
    pass


class DataSetAlreadyExistsError(DataSetError):
    """``DataSetAlreadyExistsError`` raised by ``DataCatalog`` class in case
    of trying to add a data set which already exists in the ``DataCatalog``.
    """
    pass



