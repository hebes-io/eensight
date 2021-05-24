# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os 
import glob

from typing import Union 
from pathlib import Path
from sklearn.utils import Bunch
from omegaconf import OmegaConf

from eensight.definitions import CONF_DIR, DATA_DIR, TRACK_DIR

LOCAL_FILE_URI_PREFIX = 'file:///'


def load_configuration(catalog: str, 
                       parameters: str=None, 
                       config_dir: Union[str, Path]=None, 
                       tracking_uri: str=None, 
                       save_dir: str=None) -> Bunch:
    """
    Parameters
    ----------
    catalog : str
        The name of the yaml file to use as the data catalog.
    parameters : str (default=None)
        The name of the yaml file to use for parameter configuration. If not provided, 
        defaults to `default`.
    config_dir : str or pathlib.Path object (default=`eensight.definitions.CONF_DIR`)
        The directory for the configuration files. Any valid string path is acceptable.
    tracking_uri : str (default=None)
        The address of the local or remote MLflow tracking server. If not provided, 
        defaults to `file:<save_dir>`.
    save_dir : str (default=None)
        A path to a local directory where the MLflow runs get saved. Defaults to 
        `outputs/${catalog.site_name}/mlruns` if `tracking_uri` is not provided. Has no 
        effect if `tracking_uri` is provided.

    Return
    ______
    An sklearn.utils.Bunch configuration dictionary 
    """
    cfg = Bunch()
    
    if parameters is None:
        parameters = 'default'
    
    if config_dir is None:
        config_dir = CONF_DIR
    config_dir = os.path.abspath(config_dir)
    
    # Process the catalog
    path = os.path.join(config_dir, 'catalogs')
    files = [file for file in glob.glob(path + f'/{catalog}*')]
    if not files:
        raise ValueError(f'No catalog file was found with name {catalog}')
    catalog = files[0]
    cfg.catalog = OmegaConf.load(catalog)

    if OmegaConf.is_missing(cfg.catalog, 'data_dir'):
        data_dir = DATA_DIR    
    else:
        data_dir = os.path.abspath(cfg.catalog.data_dir)
    
    cfg.data_dir = data_dir    

    # Process the parameters
    path = os.path.join(config_dir, 'parameters')
    files = [file for file in glob.glob(path + f'/{parameters}*')]
    if not files:
        raise ValueError(f'No parameter file was found with name {parameters}')
    parameters = files[0]
    cfg.parameters = OmegaConf.load(parameters)

    # Process tracking URI
    if tracking_uri is None:
        if save_dir is None:
            save_dir = os.path.join(TRACK_DIR, cfg.catalog.site_name, 'mlruns')
        
        tracking_uri = f'{LOCAL_FILE_URI_PREFIX}{save_dir}'
    cfg.tracking_uri = tracking_uri
    
    return cfg