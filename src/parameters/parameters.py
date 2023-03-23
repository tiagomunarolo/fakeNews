from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np


@dataclass
class PytorchParameter:
    # Preprocessing parameters
    model_name: str = 'cnn'
    seq_len: int = 512
    num_words: int = 30000

    # Model parameters
    embedding_size: int = 256
    out_size: int = 128
    stride: int = 2

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class XgBoostParameter:
    # Model parameters
    param_grid = dict({
        "learning_rate": [0.1, 1, 10],
        "n_estimators": [50, 100, 150, 200]
    })

    # model metadata
    model_name = 'xgboost'
    model_type = GradientBoostingClassifier


@dataclass
class SVCParameter:
    # Model parameters
    param_grid = dict({
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly',
                   'rbf', 'sigmoid', ]
    })

    # model metadata
    model_name = 'svm'
    model_type = SVC


@dataclass
class LogisticParameter:
    # Model parameters
    param_grid = dict({
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear']
    })

    # model metadata
    model_name = 'logistic'
    model_type = LogisticRegression


@dataclass
class DecisionTreeParameter:
    # Model parameters
    param_grid = dict({
        'max_depth': np.arange(10, 300, 10),
        'criterion': ['gini', 'entropy']
    })

    # model metadata
    model_name = 'decision_tree'
    model_type = DecisionTreeClassifier


@dataclass
class RandomForestParameter:
    # Model parameters
    param_grid = dict({
        'n_estimators': np.arange(2, 100, 2),
        'max_features': ['sqrt', 'log2'],
        'criterion': ["gini", "entropy", "log_loss"]
    })

    # model metadata
    model_name = 'random_forest'
    model_type = RandomForestClassifier


@dataclass
class KerasParameter:
    # Model parameters
    max_features: int = 30000  # max words in data dictionary
    pad_len: int = 512
    layer_1: int = 256
    layer_2: int = 128
    layer_3: int = 56
    epochs: int = 10
    batch_size: int = 64

    # model metadata
    model_name: str = 'lstm'
