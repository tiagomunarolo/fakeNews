from sklearn.ensemble import GradientBoostingClassifier
from models import MODELS_PATH

__MODEL_STORE_PATH__ = MODELS_PATH + "xgboost.model"
__MODEL_NAME__ = "XGBOOST"

MODEL_INFO = {
    "model_name": __MODEL_NAME__,
    "model_type": GradientBoostingClassifier,
    "store_path": __MODEL_STORE_PATH__,
    "param_grid": {
        "learning_rate": [0.1, 1, 10],
        "n_estimators": [50, 100, 150, 200]
    }
}
