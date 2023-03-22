from dataclasses import dataclass, field
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np


@dataclass
class PytorchParameter:
    # Preprocessing parameters
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
    param_grid: field(default={
        "learning_rate": [0.1, 1, 10],
        "n_estimators": [50, 100, 150, 200]
    })

    # model metadata
    model_name: str = 'xgboost'
    model_type: str = GradientBoostingClassifier


@dataclass
class SVCParameter:
    # Model parameters
    param_grid: field(default={
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly',
                   'rbf', 'sigmoid', ]
    })

    # model metadata
    model_name: str = 'svm'
    model_type: str = SVC


@dataclass
class LogisticParameter:
    # Model parameters
    param_grid: field(default={
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear', 'saga']
    })

    # model metadata
    model_name: str = 'logistic'
    model_type: str = LogisticRegression


@dataclass
class DecisionTreeParameter:
    # Model parameters
    param_grid: field(default={
        'max_depth': np.arange(10, 300, 10),
        'criterion': ['gini', 'entropy']
    })

    # model metadata
    model_name: str = 'decision_tree'
    model_type: str = DecisionTreeClassifier


@dataclass
class RandomForestParameter:
    # Model parameters
    param_grid: field(default={
        'n_estimators': np.arange(2, 100, 2),
        'max_features': ['sqrt', 'log2'],
        'criterion': ["gini", "entropy", "log_loss"]
    })

    # model metadata
    model_name: str = 'random_forest'
    model_type: str = RandomForestClassifier
