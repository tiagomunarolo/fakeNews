"""
Generic Model Classification - sklearn implementation
"""

from . import BASE_PATH
from model.base import BaseVectorizeModel
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

FOREST_PATH = BASE_PATH + "randomforest.model"
LOGISTIC_PATH = BASE_PATH + "logistic.model"
BAYES_PATH = BASE_PATH + "bayes.model"
TREE_PATH = BASE_PATH + "dtree.model"
SVM_PATH = BASE_PATH + "svm.model"

SVM_ARGS = {
    "model_type": SVC,
    "store_path": SVM_PATH,
    "param_grid": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'sigmoid', ]
    }
}

RF_ARGS = {
    "model_type": RandomForestClassifier,
    "store_path": FOREST_PATH,
    "param_grid": {
        'n_estimators': [100, 250, 500],
        'max_features': ['sqrt', 'log2'],
        'criterion': ["gini", "entropy", "log_loss"]
    }
}

LR_ARGS = {
    "model_type": LogisticRegression,
    "store_path": LOGISTIC_PATH,
    "param_grid": {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.1, 1, 10, 100],
        'solver': ['liblinear', ]
    }
}

TREE_ARGS = {
    "model_type": DecisionTreeClassifier,
    "store_path": TREE_PATH,
    "param_grid": {'criterion': ['gini', 'entropy']}
}

BAYES_ARGS = {
    "model_type": GaussianNB,
    "store_path": BAYES_PATH,
    "param_grid": {}
}

AVAILABLE_MODELS = {
    "BAYES": BAYES_ARGS,
    "SVM": SVM_ARGS,
    "DECISION_TREE": TREE_ARGS,
    "RANDOM_FOREST": RF_ARGS,
    "LINEAR": LR_ARGS,
}


class GenericModelConstructor(BaseVectorizeModel):
    """
    Generic Classification Model
    """

    @staticmethod
    def get_model_args(model: str) -> dict:
        """
        Return constructor arguments for the generic model
        :param model: model key to get its parameters
        :return:
        """
        return AVAILABLE_MODELS.get(model)

    def __init__(self, model_name: str, show_results: bool = False):
        _args = self.get_model_args(model=model_name)
        _args['show_results'] = show_results
        BaseVectorizeModel.__init__(self, **_args)