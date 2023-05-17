from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import numpy as np


@dataclass
class XgBoostParameter:
    # Model parameters
    param_grid = dict({
        "learning_rate": [0.1, 1, 10],
        "n_estimators": [50, 100, 200]
    })

    # model metadata
    model_name = 'xgboost'
    model_type = GradientBoostingClassifier


@dataclass
class SVCParameter:
    # Model parameters
    param_grid = dict({
        'C': [0.01, 0.1, 1, 10],
    })

    # model metadata
    model_name = 'svm'
    model_type = LinearSVC


@dataclass
class LogisticParameter:
    # Model parameters
    param_grid = dict({
        # l2 better
        'penalty': ['l2'],
        'C': [1e-4, 1e-3, 1e-2, 0.1, 1, 10],
        # saga performed better
        'solver': ['saga']
    })

    # model metadata
    model_name = 'logistic'
    model_type = LogisticRegression


@dataclass
class DecisionTreeParameter:
    # Model parameters
    param_grid = dict({
        'max_depth': np.arange(10, 200, 20),
        # Better than => option {gini}
        'criterion': ['entropy']
    })

    # model metadata
    model_name = 'decision_tree'
    model_type = DecisionTreeClassifier


@dataclass
class RandomForestParameter:
    # Model parameters
    param_grid = dict({
        'n_estimators': np.arange(10, 200, 20),
        'max_features': ['log2'],
        # entropy -> better performance under tests
        # option {gini, log_loss}
        'criterion': ["entropy"]
    })

    # model metadata
    model_name = 'random_forest'
    model_type = RandomForestClassifier


@dataclass
class LstmParameter:
    # Text parameters
    max_features: int = 40000  # number of != words in vocabulary
    pad_len: int = 300  # max number of words in each sentence
    epochs: int = 10
    batch_size: int = 32
    # Model parameters
    layer_1: int = 256
    layer_2: int = 128
    layer_3: int = 56
    # model metadata
    model_name: str = 'lstm'


@dataclass
class CnnParameter:
    # Text parameters
    max_features: int = 40000  # number of != words in vocabulary
    pad_len: int = 300  # max number of words in each sentence
    epochs: int = 10
    batch_size: int = 32
    transform_size: int = 100  # each word will be mapped to (1 x transform) size vector
    # model metadata
    model_name: str = 'cnn'
