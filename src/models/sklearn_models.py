"""
Base Classification Model Class - For Generic objects
SKLEARN implementations
"""
import warnings
from typing import Protocol
from sklearn.model_selection import GridSearchCV
from src.logger.logging import get_logger
from src.models.interfaces import Store

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

logger = get_logger(__file__)


class Parameter(Protocol):
    # HyperParameters of generic model
    model_name: str
    model_type: any
    param_grid: dict


class TermFrequencyClassifier:
    """
    Base Classifier Model Tf-IDF
    """

    def __init__(self, parameters: Parameter, store: Store):
        """Init Model"""
        self.store = store
        self.store.set_path(path=f"./{parameters.model_name}.model")
        self.model_type = parameters.model_type
        self.param_grid = parameters.param_grid
        self.model_name = parameters.model_name
        self.model = None
        self.tf_vector = None

    @staticmethod
    def vectorize_data(X: any, y):
        """
        Vectorize data and returns X and tf-idf object
        """
        from src.preprocess.tfidf import TfIDF
        logger.info(msg="VECTORIZE_DATA ... ")
        tf_vector = TfIDF()
        tf_vector.fit(raw_documents=X, y=y)
        return tf_vector.transform(X), tf_vector

    def fit(self, X: any, y: any, refit: bool = False) -> None:
        """
        Fit Generic provided models with GridSearchCV
        :param y: Array like, Output
        :param X: Array like, Input
        :type refit: bool: Force fit models if it no longer exists
        """
        if not refit or X is None or y is None:
            _ = self.store.read_model()
            self.__class__ = _.__class__
            self.__dict__ = _.__dict__
            return

        X, self.tf_vector = self.vectorize_data(X=X, y=y)
        estimator = self.model_type(random_state=42)
        logger.info(msg=f"FITTING_MODEL: {self.model_name} STARTED")
        grid = GridSearchCV(estimator=estimator,
                            param_grid=self.param_grid,
                            cv=5,
                            verbose=0,
                            return_train_score=False, )

        grid.fit(X=X, y=y)
        logger.info(msg=f"MODEL_FITTING: {self.model_name} DONE!")
        # select best models
        self.model = grid.best_estimator_
        logger.info(msg=f"TRAINING_SCORES :: {self.model_name} :: {grid.best_score_}")
        # Store models
        self.store.store_model(obj=self)

    def predict(self, X):
        """
        :parameter: X: Text list to be predicted
        Generates prediction, given a text
        """
        if not self.model:
            _ = self.store.read_model()
            self.__class__ = _.__class__
            self.__dict__ = _.__dict__

        X = self.tf_vector.transform(X)
        return self.model.predict(X=X)


__all__ = ['TermFrequencyClassifier', ]
