# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os 

from enum import Enum
from pathlib import Path


SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(SOURCE_DIR).resolve().parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')
WF_DIR = os.path.join(ROOT_DIR, 'workflows')


class DataStage(Enum):
    RAW             = os.path.join(DATA_DIR, '01_raw')
    INTERMEDIATE    = os.path.join(DATA_DIR, '02_intermediate')
    PRIMARY         = os.path.join(DATA_DIR, '03_primary')
    FEATURE         = os.path.join(DATA_DIR, '04_feature')



