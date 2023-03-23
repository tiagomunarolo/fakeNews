import os
from src.parameters import PytorchParameter
from src.parameters import RandomForestParameter
from src.parameters import DecisionTreeParameter
from src.parameters import SVCParameter
from src.parameters import LogisticParameter
from src.parameters import XgBoostParameter
from src.models import LstmClassifier
from src.models import TermFrequencyClassifier
from src.models import TextClassifier


class ModelNotImplementedError(Exception):
    """"""


class Executor:

    @staticmethod
    def run(model, path, refit: bool = False):
        from src.utils import get_xy_from_dataset
        X, y = get_xy_from_dataset(path=path)
        if model == 'LOGISTIC':
            TermFrequencyClassifier(parameter=LogisticParameter). \
                fit(X=X, y=y, refit=refit)
        elif model == 'XGBOOST':
            TermFrequencyClassifier(parameter=XgBoostParameter). \
                fit(X=X, y=y, refit=refit)
        elif model == 'RANDOM_FOREST':
            TermFrequencyClassifier(parameter=RandomForestParameter). \
                fit(X=X, y=y, refit=refit)
        elif model == 'SVM':
            TermFrequencyClassifier(parameter=SVCParameter) \
                .fit(X=X, y=y, refit=refit)
        elif model == 'DECISION_TREE':
            TermFrequencyClassifier(parameter=DecisionTreeParameter). \
                fit(X=X, y=y, refit=refit)
        elif model == 'LSTM':
            LstmClassifier().fit(X=X, y=y, refit=refit)
        elif model == 'CNN':
            TextClassifier(parameter=PytorchParameter). \
                fit(X=X, y=y, refit=refit)
        else:
            raise ModuleNotFoundError()


if __name__ == '__main__':
    _model_ = os.getenv("FAKE_MODEL_TRAIN", "LOGISTIC")
    _data_path_ = os.getenv('DATASET_PATH', "./data/preprocessed.csv")
    _force_ = bool(os.getenv('FORCE', False))
    Executor.run(model=_model_, path=_data_path_, refit=_force_)
