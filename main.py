import os
import numpy as np
from typing import List, Union
from src.parameters import RandomForestParameter as Rp
from src.parameters import DecisionTreeParameter as Dt
from src.parameters import SVCParameter as Svc
from src.parameters import LogisticParameter as Lr
from src.parameters import XgBoostParameter as Xp
from src.parameters import LstmParameter as Lp
from src.parameters import CnnParameter as Cp
from src.models import LstmClassifier as Lstm
from src.models import CnnClassifier as Cnn
from src.models import TermFrequencyClassifier as Tfc
from src.models.interfaces import ObjectStore as Store
from src.utils import manage_input
from src.errors import PredictionError, ModelNotImplementedError

LOGISTIC = "LOGISTIC"
XGBOOST = "XGBOOST"
RANDOM_FOREST = "RANDOM_FOREST"
SVM = "SVM"
DECISION_TREE = "DECISION_TREE"
LSTM = "LSTM"
CNN = "CNN"

MODEL = os.getenv("MODEL", "LOGISTIC")
DATA = os.getenv('DATA', "./data/preprocessed.csv")
FORCE = bool(os.getenv('FORCE', False))
PREDICT = bool(os.getenv('PREDICT', True))


class Executor:

    @staticmethod
    def run(model, path: str, refit: bool = False):
        """
        Run --> execute model
        Parameters
        ----------
        model:
        path: str :: path
        refit: bool :: force models to be refitted
        """
        from src.utils import get_xy_from_dataset
        X, y = get_xy_from_dataset(path=path)
        if model == LOGISTIC:
            Tfc(parameters=Lr, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == XGBOOST:
            Tfc(parameters=Xp, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == RANDOM_FOREST:
            Tfc(parameters=Rp, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == SVM:
            Tfc(parameters=Svc, store=Store()) \
                .fit(X=X, y=y, refit=refit)
        elif model == DECISION_TREE:
            Tfc(parameters=Dt, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == LSTM:
            Lstm(parameters=Lp, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == CNN:
            Cnn(parameters=Cp, store=Store()). \
                fit(X=X, y=y, refit=refit)
        else:
            raise ModelNotImplementedError


class Predictor:

    @staticmethod
    def predict(X: Union[str, List[str]]):
        if not X:
            raise PredictionError
        X = manage_input(text=X)
        p0 = Tfc(parameters=Lr, store=Store()).predict(X=X)
        p1 = Tfc(parameters=Xp, store=Store()).predict(X=X)
        p2 = Tfc(parameters=Rp, store=Store()).predict(X=X)
        p3 = Tfc(parameters=Svc, store=Store()).predict(X=X)
        p4 = Tfc(parameters=Dt, store=Store()).predict(X=X)
        p5 = Lstm(parameters=Lp, store=Store()).predict(X=X)
        p6 = Cnn(parameters=Cp, store=Store()).predict(X=X)
        predictions_ = [p0[0], p1[0], p2[0], p3[0], p4[0], p5[0][0], p6]
        final_response = np.bincount(predictions_).argmax()
        print(final_response == 1)


if __name__ == 'main':
    if PREDICT:
        Predictor.predict(X=input())
    else:
        Executor.run(model=MODEL, path=DATA, refit=FORCE)
