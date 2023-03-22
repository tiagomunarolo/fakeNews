"""
Base Classification Model Class - For Generic objects
SKLEARN implementations
"""
import os.path
import os
import pickle
import warnings
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from models.logger import get_logger

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

logger = get_logger(__file__)

__all__ = [
    'ObjectStore',
    'TermFrequencyClassifier'
]


@dataclass
class ObjectStore:
    """
    Generic Object Store Class
    """
    store_path: str

    @property
    def path_exists(self) -> bool:
        """
        Check if stor_path exists
        :return:
        """
        return os.path.exists(self.store_path)

    def store_model(self, obj) -> None:
        """
        Save models to ./models dir
        """
        with open(self.store_path, 'wb') as file:
            logger.info(msg=f"STORING_MODEL: {self.store_path}")
            pickle.dump(obj=obj, file=file)
            logger.info(msg=f"MODEL_STORED: {self.store_path}")

    def read_model(self):
        """
        Reads Stored models from ./models dir
        superclass
        """
        logger.info(msg=f"READING_MODEL: {self.store_path}")
        if not self.path_exists:
            raise FileNotFoundError(f'{self.store_path} does not exists')
        with open(self.store_path, 'rb') as file:
            class_model = pickle.load(file=file)
            logger.info(msg=f"MODEL_LOADED: {self.store_path} COMPLETED")
            return class_model


class TermFrequencyClassifier(ObjectStore):
    """
    Base Classifier Model Tf-IDF
    """

    def __init__(self, **kwargs):
        """
        Mandatory args:
        - model_type: str: sklearn models to be fit
        - store_path: str:  Where models should be stored or read
        :param kwargs:
        """
        super().__init__(store_path=kwargs['store_path'])
        self.model_type = kwargs['model_type']
        self.show_results = kwargs.get('show_results', False)
        self.param_grid = kwargs.get('param_grid', {})
        self.model_name = kwargs['model_name']
        self.model = None
        self.tf_vector = None

    @staticmethod
    def vectorize_data(X: any):
        """
        Vectorize dataset and returns X and tf-idf object
        """
        logger.info(msg="vectorize_data started")
        tf_vector = TfidfVectorizer()
        tf_vector.fit(X)
        logger.info(msg="vectorize_data finished")
        return tf_vector.transform(X), tf_vector

    def fit(self, X: any, y: any, force: bool = False):
        """
        Fit Generic provided models with GridSearchCV
        :param y: Array like, Output
        :param X: Array like, Input
        :type force: bool: Force fit models if it no longer exists
        """
        if not force and self.path_exists:
            clf = self.read_model()
            self.model = clf.model
            self.tf_vector = clf.tf_vector
            return self

        if not force and (not self.path_exists or None in [X, y]):
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
