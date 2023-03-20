"""
__init__ declarations
"""
import os

__PROJECT_NAME__ = 'fakenews'
__FILE_PATH__ = os.path.abspath(__file__)
ROOT_PATH = __FILE_PATH__.split(__PROJECT_NAME__)[0] + __PROJECT_NAME__

MODELS_PATH = f"{ROOT_PATH}/model/models/"
DATASET_PATH = f"{ROOT_PATH}/dataset/"