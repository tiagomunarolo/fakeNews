"""
Logistic Regression Model - sklearn implementation
"""
from model.base import Base
from . import BASE_PATH
from sklearn.linear_model import LogisticRegression

LOGISTIC_PATH = BASE_PATH + "/logistic.pyc"


class LogisticModel(Base):
    """
    Logistic Regression Model
    """

    MODEL_ARGS = {
        "show_results": False,
        "model_type": LogisticRegression,
        "store_path": LOGISTIC_PATH,
    }

    def __init__(self, show_results=False):
        if show_results:
            LogisticModel.MODEL_ARGS['show_results'] = True
        Base.__init__(self, **LogisticModel.MODEL_ARGS)