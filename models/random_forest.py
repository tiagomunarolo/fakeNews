import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models import MODELS_PATH

__MODEL_STORE_PATH__ = MODELS_PATH + "rf.model"
__MODEL_NAME__ = "RANDOM_FOREST"

MODEL_INFO = {
    "model_name": __MODEL_NAME__,
    "model_type": RandomForestClassifier,
    "store_path": __MODEL_STORE_PATH__,
    "param_grid": {
        'n_estimators': np.arange(2, 100, 2),
        'max_features': ['sqrt', 'log2'],
        'criterion': ["gini", "entropy", "log_loss"]
    }
}
