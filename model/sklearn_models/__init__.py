"""
__init__ file
"""
import os

__PROJECT_NAME__ = 'fakeNews'
__MODELS__ = 'model/save_models/'
__DATASET__ = "dataset"
__FILE_PATH__ = os.path.abspath(__file__)
ROOT_PATH = __FILE_PATH__.split(__PROJECT_NAME__)[0] + __PROJECT_NAME__

MODELS_PATH = f"{ROOT_PATH}/{__MODELS__}"
DATASET_PATH = f"{ROOT_PATH}/{__DATASET__}/"

if not os.path.exists(path=MODELS_PATH):
    os.makedirs(MODELS_PATH, exist_ok=True)

__all__ = ['MODELS_PATH', 'DATASET_PATH']
