import os
from src.parameters import PytorchParameter
from src.parameters import RandomForestParameter
from src.parameters import DecisionTreeParameter
from src.parameters import SVCParameter
from src.parameters import LogisticParameter
from src.parameters import XgBoostParameter
from src.parameters import KerasParameter
from src.models import LstmClassifier
from src.models import TermFrequencyClassifier as TFClassifier
from src.models import CNNClassifier
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
        store = Store()
        X, y = get_xy_from_dataset(path=path)
        if model == __LOGISTIC__:
            TFClassifier(parameters=LogisticParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == __XGBOOST__:
            TFClassifier(parameters=XgBoostParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == __RANDOM_FOREST__:
            TFClassifier(parameters=RandomForestParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == __SVM__:
            TFClassifier(parameters=SVCParameter, store=store) \
                .fit(X=X, y=y, refit=refit)
        elif model == __DECISION_TREE__:
            TFClassifier(parameters=DecisionTreeParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == __LSTM__:
            LstmClassifier(parameters=KerasParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == __CNN__:
            CNNClassifier(parameters=PytorchParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        else:
            raise ModelNotImplementedError


class Predictor:

    @staticmethod
    def predict(x: str):
        if not x:
            raise PredictionError
        # store = Store()
        # X = manage_input(text=x)
        # p0 = TFClassifier(parameters=LogisticParameter, store=store).predict(X=X)
        # p1 = TFClassifier(parameters=XgBoostParameter, store=store).predict(X=X)
        # p2 = TFClassifier(parameters=RandomForestParameter, store=store).predict(X=X)
        # p3 = TFClassifier(parameters=SVCParameter, store=store).predict(X=X)
        # p4 = TFClassifier(parameters=DecisionTreeParameter, store=store).predict(X=X)
        # p5 = LstmClassifier(parameters=KerasParameter, store=store).predict(X=X)
        # prediction = CNNClassifier(PytorchParameter, store).predict(X=X)
        # predictions_ = [p0[0], p1[0], p2[0], p3[0], p4[0], p5[0][0]]
        # print(predictions_)
        # final_response = np.bincount(predictions_).argmax()
        # print(final_response)
        # print(prediction)


if __name__ == '__main__':
    # Execute Prediction
    if __PREDICT__:
        Predictor.predict("<TEXT>")
    else:
        # Train model provided
        Executor.run(model=__RUN_MODEL__, path=__DATA__, refit=__FORCE__)
