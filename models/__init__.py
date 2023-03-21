"""
Models implementations for Classification Binary Tasks
1- Sklearn models
2- Keras models
"""
import os

__PROJECT_NAME__ = 'fakeNews'
__FILE_PATH__ = os.path.abspath(__file__)
ROOT_PATH = __FILE_PATH__.split(__PROJECT_NAME__)[0] + __PROJECT_NAME__
MODELS_PATH = f"{ROOT_PATH}/models/save_models/"

if not os.path.exists(path=MODELS_PATH):
    os.makedirs(MODELS_PATH, exist_ok=True)
