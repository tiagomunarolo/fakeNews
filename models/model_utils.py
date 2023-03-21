"""
Utils file
"""
import os
import pandas as pd
from typing import Tuple
from models.base import TermFrequencyClassifier
from models.linear_model import MODEL_INFO as LINEAR
from models.xgboost import MODEL_INFO as XGBOOST
from models.decision_tree import MODEL_INFO as TREE
from models.random_forest import MODEL_INFO as RFOREST
from models.svm import MODEL_INFO as SVM

__all__ = ['get_xy_from_dataset',
           'ALL_MODELS',
           'execute_classifier']

ALL_MODELS = {
    "LINEAR": LINEAR,
    "DECISION_TREE": TREE,
    "XGBOOST": XGBOOST,
    "RANDOM_FOREST": RFOREST,
    "SVM": SVM,
}


def get_xy_from_dataset(path: str | None = None) \
        -> Tuple[pd.Series, pd.Series]:
    """
    Reads Training Dataset
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found!')

    data = pd.read_csv(path, index_col=0)
    X = data.TEXT
    Y = data.LABEL
    return X, Y


def execute_classifier(X, y, model_info, force=False):
    """

    Parameters
    ----------
    X
    y
    model_info
    force
    """
    linear = TermFrequencyClassifier(**model_info)
    linear.fit(X, y, force=force)
