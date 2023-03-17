import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

PROJECT_PATH = os.getenv('PROJECT_PATH', None)

assert PROJECT_PATH, "PROJECT_PATH ENV MUST BE SET"
MODELS_PATH = f"{PROJECT_PATH}/model/models/"
DATASET_PATH = f"{PROJECT_PATH}/dataset/final_dataset.csv"

FOREST_PATH = MODELS_PATH + "randomforest.model"
LOGISTIC_PATH = MODELS_PATH + "logistic.model"
BAYES_PATH = MODELS_PATH + "bayes.model"
TREE_PATH = MODELS_PATH + "dtree.model"
SVM_PATH = MODELS_PATH + "svm.model"

SVM_ARGS = {
    "model_type": SVC,
    "store_path": SVM_PATH,
    "param_grid": {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
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

BAYES_ARGS = {
    "model_type": GaussianNB,
    "store_path": BAYES_PATH,
    "param_grid": {}
}

AVAILABLE_MODELS = {
    # "BAYES": BAYES_ARGS,
    "LINEAR": LR_ARGS,
    "DECISION_TREE": TREE_ARGS,
    "RANDOM_FOREST": RF_ARGS,
    "SVM": SVM_ARGS,
}