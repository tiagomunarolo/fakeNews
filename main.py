from typing import List, Union, Dict
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

    @staticmethod
    def predict(X: Union[str, List[str]],
                model: Union[str, List[str]] = "") -> \
            Dict[str, Union[bool, List[bool]]]:

        if model and isinstance(model, str) and model not in MODELS:
            raise ModelNotImplementedError(f"{model} Not Implemented!")
        elif isinstance(model, str):
            execution_list = [model]
        elif isinstance(model, list):
            execution_list = model
        else:
            execution_list = MODELS

        predictions = dict()
        for model in execution_list:
            param, classifier = CLASSIFIER_PARAMETERS.get(model)
            response = classifier(parameters=param).predict(X=X)
            predictions[model] = response

        return predictions
