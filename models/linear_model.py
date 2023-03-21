from sklearn.linear_model import LogisticRegression
from models import MODELS_PATH

__MODEL_STORE_PATH__ = MODELS_PATH + "linear.model"
__MODEL_NAME__ = "LINEAR"

MODEL_INFO = {
    "model_name": __MODEL_NAME__,
    "model_type": LogisticRegression,
    "store_path": __MODEL_STORE_PATH__,
    "param_grid": {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear', 'saga']
    }
}
