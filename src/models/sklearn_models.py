"""
Base Classification Model Class - For Generic objects
SKLEARN implementations
"""
import os.path
import os
import pickle
import warnings
from typing import Protocol
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from src.logger.logging import get_logger

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

logger = get_logger(__file__)

__all__ = [
    'ObjectStore',
    'TermFrequencyClassifier',
]


@dataclass
class ObjectStore:
    """
    Generic Object Store Class
    """
    path: str

    @property
    def path_exists(self) -> bool:
        """
        Check if path exists
        :return:
        """
        return os.path.exists(self.path)

    def store_model(self, obj) -> None:
        """
        Save models to ./models dir
        """
        with open(self.path, 'wb') as file:
            logger.info(msg=f"STORING_MODEL: {self.path}")
            pickle.dump(obj=obj, file=file)
            logger.info(msg=f"MODEL_STORED: {self.path}")

    def read_model(self):
        """
        Reads Stored model from provided dir
        superclass
        """
        logger.info(msg=f"READING_MODEL: {self.path}")
        if not self.path_exists:
            raise FileNotFoundError(f'{self.path} does not exists')
        with open(self.path, 'rb') as file:
            class_model = pickle.load(file=file)
            logger.info(msg=f"MODEL_LOADED: {self.path} COMPLETED")
            return class_model


class Parameter(Protocol):
    # HyperParameters of generic model
    model_name: str
    model_type: any
    param_grid: dict


class TermFrequencyClassifier(ObjectStore):
    """
    Base Classifier Model Tf-IDF
    """

    def __init__(self, parameter: Parameter):
        """Init Model"""
        super().__init__(path=f"./{parameter.model_name}.model")
        self.model_type = parameter.model_type
        self.param_grid = parameter.param_grid
        self.model_name = parameter.model_name
        self.model = None
        self.tf_vector = None

    @staticmethod
    def vectorize_data(X: any):
        """
        Vectorize data and returns X and tf-idf object
        """
        logger.info(msg="vectorize_data started")
        tf_vector = TfidfVectorizer()
        tf_vector.fit(X)
        logger.info(msg="vectorize_data finished")
        return tf_vector.transform(X), tf_vector

    def fit(self, X: any, y: any, refit: bool = False):
        """
        Fit Generic provided models with GridSearchCV
        :param y: Array like, Output
        :param X: Array like, Input
        :type refit: bool: Force fit models if it no longer exists
        """
        if not refit and self.path_exists:
            clf = self.read_model()
            self.model = clf.model
            self.tf_vector = clf.tf_vector
            return self

        if not refit and (not self.path_exists or None in [X, y]):
            raise Exception("MODEL_NOT_FITTED_YET")

        logger.info(msg=f"Fitting models {self.model_name}")
        X, self.tf_vector = self.vectorize_data(X=X)
        if self.model_name == 'VOTING':
            estimator = self.model_type
        else:
            estimator = self.model_type(random_state=42)
        grid = GridSearchCV(estimator=estimator,
                            param_grid=self.param_grid,
                            cv=3,
                            verbose=5, )

        grid.fit(X=X, y=y)
        logger.info(msg=f"Model fitted {self.model_name}")
        # select best models
        self.model = grid.best_estimator_
        # Store models
        self.store_model(obj=self)
        return self

    def predict(self, X):
        """
        :parameter: X: Text list to be predicted
        Generates prediction, given a text
        """
        from crawler.dataset_builder import generate_dataset_for_input
        if not self.model:
            # Restore information stored
            self.fit(None, None)

        X = generate_dataset_for_input(df=X)
        X = self.tf_vector.transform(X)
        return self.model.predict(X=X)
