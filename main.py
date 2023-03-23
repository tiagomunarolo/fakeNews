import os
from src.parameters import PytorchParameter
from src.parameters import RandomForestParameter
from src.parameters import DecisionTreeParameter
from src.parameters import SVCParameter
from src.parameters import LogisticParameter
from src.parameters import XgBoostParameter
from src.parameters import KerasParameter
from src.models import LstmClassifier
from src.models import TermFrequencyClassifier
from src.models import TextClassifier
from src.models.interfaces import ObjectStore as Store


class ModelNotImplementedError(Exception):
    """"""


class Executor:

    @staticmethod
    def run(model, path, refit: bool = False):
        from src.utils import get_xy_from_dataset
        store = Store()
        X, y = get_xy_from_dataset(path=path)
        if model == 'LOGISTIC':
            TermFrequencyClassifier(parameters=LogisticParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == 'XGBOOST':
            TermFrequencyClassifier(parameters=XgBoostParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == 'RANDOM_FOREST':
            TermFrequencyClassifier(parameters=RandomForestParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == 'SVM':
            TermFrequencyClassifier(parameters=SVCParameter, store=store) \
                .fit(X=X, y=y, refit=refit)
        elif model == 'DECISION_TREE':
            TermFrequencyClassifier(parameters=DecisionTreeParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == 'LSTM':
            LstmClassifier(parameters=KerasParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        elif model == 'CNN':
            TextClassifier(parameters=PytorchParameter, store=store). \
                fit(X=X, y=y, refit=refit)
        else:
            raise ModelNotImplementedError


if __name__ == '__main__':
    _model_ = os.getenv("FAKE_MODEL_TRAIN", "LOGISTIC")
    _data_path_ = os.getenv('DATASET_PATH', "./data/preprocessed.csv")
    _force_ = bool(os.getenv('FORCE', False))
    Executor.run(model=_model_, path=_data_path_, refit=_force_)
