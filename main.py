import os
import numpy as np
from src.parameters import RandomForestParameter
from src.parameters import DecisionTreeParameter
from src.parameters import SVCParameter
from src.parameters import LogisticParameter
from src.parameters import XgBoostParameter
from src.parameters import KerasParameter
from src.models import LstmClassifier
from src.models import TermFrequencyClassifier as TFClassifier
from src.models.interfaces import ObjectStore as Store
from src.utils import manage_input
from src.errors import PredictionError, ModelNotImplementedError

__LOGISTIC__ = "LOGISTIC"
__XGBOOST__ = "XGBOOST"
__RANDOM_FOREST__ = "RANDOM_FOREST"
__SVM__ = "SVM"
__DECISION_TREE__ = "DECISION_TREE"
__LSTM__ = "LSTM"
__CNN__ = "CNN"

__RUN_MODEL__ = os.getenv("MODEL", "LOGISTIC")
__DATA__ = os.getenv('DATA', "./data/preprocessed.csv")
__FORCE__ = bool(os.getenv('FORCE', False))
__PREDICT__ = bool(os.getenv('PREDICT', True))


class Executor:

    @staticmethod
    def run(model, path, refit: bool = False):
        from src.utils import get_xy_from_dataset
        X, y = get_xy_from_dataset(path=path)
        if model == __LOGISTIC__:
            TFClassifier(parameters=LogisticParameter, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == __XGBOOST__:
            TFClassifier(parameters=XgBoostParameter, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == __RANDOM_FOREST__:
            TFClassifier(parameters=RandomForestParameter, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == __SVM__:
            TFClassifier(parameters=SVCParameter, store=Store()) \
                .fit(X=X, y=y, refit=refit)
        elif model == __DECISION_TREE__:
            TFClassifier(parameters=DecisionTreeParameter, store=Store()). \
                fit(X=X, y=y, refit=refit)
        elif model == __LSTM__:
            LstmClassifier(parameters=KerasParameter, store=Store()). \
                fit(X=X, y=y, refit=refit)
        # elif model == __CNN__:
        #     CNNClassifier(parameters=PytorchParameter, store=Store()). \
        #         fit(X=X, y=y, refit=refit)
        else:
            raise ModelNotImplementedError


class Predictor:

    @staticmethod
    def predict(X: str):
        if not X:
            raise PredictionError
        X = manage_input(text=X)

        p0 = TFClassifier(parameters=LogisticParameter, store=Store()).predict(X=X)
        p1 = TFClassifier(parameters=XgBoostParameter, store=Store()).predict(X=X)
        p2 = TFClassifier(parameters=RandomForestParameter, store=Store()).predict(X=X)
        p3 = TFClassifier(parameters=SVCParameter, store=Store()).predict(X=X)
        p4 = TFClassifier(parameters=DecisionTreeParameter, store=Store()).predict(X=X)
        p5 = LstmClassifier(parameters=KerasParameter, store=Store()).predict(X=X)
        # p6 = CNNClassifier(parameters=PytorchParameter, store=Store()).predict(X=X)
        predictions_ = [p0[0], p1[0], p2[0], p3[0], p4[0], p5[0][0]]
        final_response = np.bincount(predictions_).argmax()
        print(final_response == 1)


if __name__ == '__main__':
    if __PREDICT__:
        text = input()
        Predictor.predict(X=text)
    else:
        # Train model provided
        Executor.run(model=__RUN_MODEL__, path=__DATA__, refit=__FORCE__)
