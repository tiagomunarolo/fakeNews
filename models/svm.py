from sklearn.svm import SVC
from models import MODELS_PATH

__MODEL_STORE_PATH__ = MODELS_PATH + "svm.model"
__MODEL_NAME__ = "SVM"

MODEL_INFO = {
    "model_name": __MODEL_NAME__,
    "model_type": SVC,
    "store_path": __MODEL_STORE_PATH__,
    "param_grid": {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', ]
    }
}
