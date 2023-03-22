"""
__init__ declarations
"""
import os

__PROJECT_NAME__ = 'fakeNews'
__FILE_PATH__ = os.path.abspath(__file__)
ROOT_PATH = __FILE_PATH__.split(__PROJECT_NAME__)[0] + __PROJECT_NAME__
DATASET_PATH = f"{ROOT_PATH}/data/"

if not os.path.exists(path=DATASET_PATH):
    os.makedirs(DATASET_PATH, exist_ok=True)
