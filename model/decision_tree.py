"""
Decision Tree Model - sklearn implementation
"""
from . import BASE_PATH
from model.base import Base
from sklearn.tree import DecisionTreeClassifier

TREE_PATH = BASE_PATH + "/dtree.pyc"


class DecisionTree(Base):
    """
    Decision Tree Classification Model
    """
    MODEL_ARGS = {
        "show_results": False,
        "model_type": DecisionTreeClassifier,
        "store_path": TREE_PATH,
    }

    def __init__(self, show_results=False):
        if show_results:
            DecisionTree.MODEL_ARGS['show_results'] = True
        Base.__init__(self, **DecisionTree.MODEL_ARGS)