import os 
from pathlib import Path

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(SOURCE_DIR).resolve().parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')


