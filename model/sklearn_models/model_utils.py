"""
Defines models arguments and its hyper parameters
For now:
1- sklearn.ensemble.RandomForestClassifier
2- sklearn.tree.DecisionTreeClassifier
3- sklearn.linear_model.LogisticRegression
4- sklearn.ensemble.GradientBoostingClassifier
5- sklearn.svm.SVC
"""
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from typing import Tuple
from . import MODELS_PATH

FOREST_PATH = MODELS_PATH + "randomforest.model"
LOGISTIC_PATH = MODELS_PATH + "logistic.model"
XG_PATH = MODELS_PATH + "xgboost.model"
TREE_PATH = MODELS_PATH + "dtree.model"
SVM_PATH = MODELS_PATH + "svm.model"

SVM_ARGS = {
    "model_type": SVC,
    "store_path": SVM_PATH,
    "param_grid": {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', ]
    }
}

RF_ARGS = {
    "model_type": RandomForestClassifier,
    "store_path": FOREST_PATH,
    "param_grid": {
        'n_estimators': np.arange(2, 100, 2),
        'max_features': ['sqrt', 'log2'],
        'criterion': ["gini", "entropy", "log_loss"]
    }
}

LR_ARGS = {
    "model_type": LogisticRegression,
    "store_path": LOGISTIC_PATH,
    "param_grid": {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    }
}

TREE_ARGS = {
    "model_type": DecisionTreeClassifier,
    "store_path": TREE_PATH,
    "param_grid": {
        'max_depth': np.arange(2, 100, 2),
        'criterion': ['gini', 'entropy']

    }
}

XGBOOST_ARGS = {
    "model_type": GradientBoostingClassifier,
    "store_path": XG_PATH,
    "param_grid": {
        "learning_rate": [0.1, 0.25, 0.5, 0.75, 1],
        "n_estimators": [10, 100, 150, 200]
    }
}

AVAILABLE_MODELS = {
    "LINEAR": LR_ARGS,
    "DECISION_TREE": TREE_ARGS,
    "RANDOM_FOREST": RF_ARGS,
    "SVM": SVM_ARGS,
    "XGBOOST": XGBOOST_ARGS
}


def get_xy_from_dataset(path: str | None = None) -> Tuple[pd.Series, pd.Series]:
    """
    Reads Training Dataset
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found!')

    data = pd.read_csv(path, index_col=0)
    X = data.TEXT
    Y = data.LABEL
    return X, Y