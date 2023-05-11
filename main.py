from typing import List, Union
from src.models import LstmClassifier
from src.models import CnnClassifier
from src.models import TfClassifier
from src.errors import ModelNotImplementedError
from src.preprocess import get_dataset
from src.parameters import *
from src.constants import *

DATA = os.getenv('DATA', "./data/preprocessed.csv")

CLASSIFIER_PARAMETERS = {
    LOGISTIC: LogisticParameter,
    XGBOOST: XgBoostParameter,
    RANDOM_FOREST: RandomForestParameter,
    CNN: CnnParameter,
    LSTM: LstmParameter,
    DECISION_TREE: DecisionTreeParameter,
    SVM: SVCParameter
}


class Executor:

    @staticmethod
    def run(model, path: str = DATA, refit: bool = False):
        """
        Run --> execute model
        Parameters
        ----------
        model:
        path: str :: path of data training
        refit: bool :: force models to be refitted
        """
        X, y = get_dataset(path=path)
        if model not in MODELS:
            raise ModelNotImplementedError(f"{model} Not Implemented!")
        elif model == CNN:
            classifier = CnnClassifier
        elif model == LSTM:
            classifier = LstmClassifier
        else:
            classifier = TfClassifier

        param = CLASSIFIER_PARAMETERS.get(model)
        classifier(parameters=param).fit(X=X, y=y, refit=refit)


class Predictor:

    @staticmethod
    def predict(X: Union[str, List[str]]):
        ...
