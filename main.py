from typing import List, Union
from src.models import LstmClassifier
from src.models import CnnClassifier
from src.models import TfClassifier
from src.errors import ModelNotImplementedError
from src.preprocess import load_dataset
from src.parameters import *
from src.constants import *

DATA = os.getenv('DATA', "./data/preprocessed.csv")

CLASSIFIER_PARAMETERS = {
    LOGISTIC: (LogisticParameter, TfClassifier),
    XGBOOST: (XgBoostParameter, TfClassifier),
    RANDOM_FOREST: (RandomForestParameter, TfClassifier),
    DECISION_TREE: (DecisionTreeParameter, TfClassifier),
    SVM: (SVCParameter, TfClassifier),
    CNN: (CnnParameter, CnnClassifier),
    LSTM: (LstmParameter, LstmClassifier),
}


class Executor:

    @staticmethod
    def fit(model: str, path: str = DATA, refit: bool = False):
        """
        Run --> Fit <model> classifier
        Parameters
        ----------
        model: str :: Model identifier
        path: str :: path of data training
        refit: bool :: force models to be refitted
        """
        X, y = load_dataset(path=path)
        if model not in MODELS:
            raise ModelNotImplementedError(f"{model} Not Implemented!")

        param, classifier = CLASSIFIER_PARAMETERS.get(model)
        classifier(parameters=param).fit(X=X, y=y, refit=refit)


class Predictor:

    @staticmethod
    def predict(X: Union[str, List[str]]):
        ...
