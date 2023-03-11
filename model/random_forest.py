"""
Random Forest Model - sklearn implementation
"""
from . import BASE_PATH
from model.base import Base
from sklearn.ensemble import RandomForestClassifier

FOREST_PATH = BASE_PATH + "/randomforest.pyc"


class RandomForestModel(Base):
    """
    Random Forest Classification Model
    """

    MODEL_ARGS = {
        "show_results": False,
        "model_type": RandomForestClassifier,
        "store_path": FOREST_PATH,
        "param_grid": {
            'n_estimators': [100, 200, 500, 1000],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    }

    def __init__(self, show_results=False):
        if show_results:
            RandomForestModel.MODEL_ARGS['show_results'] = True
        Base.__init__(self, **RandomForestModel.MODEL_ARGS)