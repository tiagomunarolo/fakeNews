import numpy as np
from sklearn.tree import DecisionTreeClassifier
from models import MODELS_PATH

__MODEL_STORE_PATH__ = MODELS_PATH + "dtree.model"
__MODEL_NAME__ = "DECISION_TREE"

MODEL_INFO = {
    "model_name": __MODEL_NAME__,
    "model_type": DecisionTreeClassifier,
    "store_path": __MODEL_STORE_PATH__,
    "param_grid": {
        'max_depth': np.arange(10, 300, 10),
        'criterion': ['gini', 'entropy']

    }
}
