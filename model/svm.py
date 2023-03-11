"""
SVM Model - sklearn implementation
"""
from . import BASE_PATH
from model.base import Base
from sklearn.svm import SVC

SVM_PATH = BASE_PATH + "/svm.pyc"


class SVMModel(Base):
    """
    SVC Regression Model
    """

    MODEL_ARGS = {
        "show_results": False,
        "model_type": SVC,
        "store_path": SVM_PATH,
        "param_grid": {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']}
    }

    def __init__(self, show_results=False):
        if show_results:
            SVMModel.MODEL_ARGS['show_results'] = True
        Base.__init__(self, **SVMModel.MODEL_ARGS)