"""
Bayes Model - sklearn implementation
"""
from . import BASE_PATH
from model.base import Base
from sklearn.naive_bayes import GaussianNB

BAYES_PATH = BASE_PATH + "/bayes.pyc"


class BayesModel(Base):
    """
    Bayes Classification Model
    """
    MODEL_ARGS = {
        "show_results": False,
        "model_type": GaussianNB,
        "store_path": BAYES_PATH,
    }

    def __init__(self, show_results=False):
        if show_results:
            BayesModel.MODEL_ARGS['show_results'] = True
        Base.__init__(self, **BayesModel.MODEL_ARGS)