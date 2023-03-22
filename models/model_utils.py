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
from models.random_forest import MODEL_INFO as R_FOREST
from models.svm import MODEL_INFO as SVM
from models.voting_model import MODEL_INFO as VOTING

ALL_MODELS = {
    "LINEAR": LINEAR,
    "DECISION_TREE": TREE,
    "XGBOOST": XGBOOST,
    "RANDOM_FOREST": R_FOREST,
    "SVM": SVM,
    "VOTING": VOTING
}


def get_xy_from_dataset(path: str = "") \
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
    model = TermFrequencyClassifier(**model_info)
    model.fit(X, y, force=force)
    return model


def get_prediction(X, model_info):
    """
    :parameter: X: Text list to be predicted
    Generates prediction, given a text
    """
    # To read stored model only
    clf = execute_classifier(None, None, model_info=model_info)
    return clf.predict(X=X)
