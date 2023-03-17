"""
__init__ file
"""
import os
from .base import GenericStoreModel, BaseTermFrequencyModel, \
    AVAILABLE_MODELS, GenericModelConstructor

PROJECT_PATH = os.getenv('PROJECT_PATH', None)

assert PROJECT_PATH, "PROJECT_PATH ENV MUST BE SET"
MODELS_PATH = f"{PROJECT_PATH}/model/models/"

__all__ = [
    'GenericStoreModel',
    'BaseTermFrequencyModel',
    'AVAILABLE_MODELS',
    'GenericModelConstructor',
    'PROJECT_PATH',
    'MODELS_PATH',
]