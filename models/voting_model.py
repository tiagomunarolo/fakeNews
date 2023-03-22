""""
Voting Classifier Implementation - SkLearn lib
"""
import os
import pickle
from mlxtend.classifier import EnsembleVoteClassifier
from models import MODELS_PATH

__MODEL_STORE_PATH__ = MODELS_PATH + "voting.model"
__MODEL_NAME__ = "VOTING"

_estimators = []
for file in os.listdir(path=MODELS_PATH):
    path = MODELS_PATH + file
    if not path.endswith(".model") or path == __MODEL_STORE_PATH__:
        continue
    if os.path.exists(path=path):
        with open(path, 'rb') as f:
            _ = pickle.load(file=f)
            _estimators.append(_.model)

MODEL_INFO = {
    "model_name": __MODEL_NAME__,
    "model_type": EnsembleVoteClassifier(
        clfs=_estimators,
        fit_base_estimators=False,
        verbose=2),
    "store_path": __MODEL_STORE_PATH__,
    "param_grid": {},
}
