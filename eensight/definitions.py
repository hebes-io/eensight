# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os 

from enum import Enum
from pathlib import Path


SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(SOURCE_DIR).resolve().parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CONF_DIR = os.path.join(ROOT_DIR, 'conf')
TRACK_DIR = os.path.join(ROOT_DIR, 'outputs')


# The data stage according to Kedro's data engineering convention
# https://kedro.readthedocs.io/en/stable/12_faq/01_faq.html#what-is-data-engineering-convention

class DataStage(Enum):
    NONE            = ''
    RAW             = '01_raw'
    INTERMEDIATE    = '02_intermediate'
    PRIMARY         = '03_primary'
    FEATURE         = '04_feature'
    ML_INPUT        = '05_model_input'
    ML_OUTPUT       = '06_model_output'







