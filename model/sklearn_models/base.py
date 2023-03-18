"""
Base Classification Model Class - For Generic objects
SKLEARN implementations
"""
import os.path
import os
import pickle
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from model.sklearn_models.model_utils import AVAILABLE_MODELS
from typing import Tuple
import pandas as pd
import warnings
import logging

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)


def get_xy_from_dataset(path: str | None = None) -> Tuple[pd.Series, pd.Series]:
    """
    Reads Training Dataset
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found!')

    data = pd.read_csv(path, index_col=0)
    X = data.TEXT
    Y = data.LABEL
    return X, Y


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name="BASE_MODEL")


@dataclass(slots=True)
class GenericStoreModel:
    """
    Generic Store Model
    """
    store_path: str

    @property
    def path_exists(self) -> bool:
        """
        Check if stor_path exists
        :return:
        """
        return os.path.exists(self.store_path)

    def _store_model(self, obj) -> None:
        """
        Save model to ./models dir
        """
        with open(self.store_path, 'wb') as file:
            logger.info(msg=f"STORING MODEL: {self.store_path}")
            pickle.dump(obj=obj, file=file)

    def _read_model(self, model_only: bool = True):
        """
        Read GenericModel from ./models dir
        :type model_only: bool: if True return sklearn model stored only
        """
        logger.info(msg=f"READING MODEL: {self.store_path}")
        if not self.path_exists:
            raise FileNotFoundError(f'{self.store_path} does not exists')
        with open(self.store_path, 'rb') as file:
            class_model = pickle.load(file=file)
            if model_only:
                return class_model.model
            else:
                return class_model


class BaseTermFrequencyModel(GenericStoreModel):
    """
    Base Classifier Model Tf-IDF
    """

    def __init__(self, **kwargs):
        """
        Mandatory args:
        - model_type: str: sklearn model to be fit
        - store_path: str:  Where model should be stored or read
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

    def fit(self, X: any, y: any, force: bool = False) -> None:
        """
        Fit Generic provided model with GridSearchCV
        :param y: Array like
        :param X: Array like
        :type force: bool: Force fit model if it no longer exists
        """
        if not force and self.path_exists:
            self.model = self._read_model()
            return

        logger.info(msg=f"Fitting model {self.model_name}")
        X, self.tf_vector = self.vectorize_data(X=X)
        estimator = self.model_type(random_state=42)
        grid = GridSearchCV(estimator=estimator,
                            param_grid=self.param_grid,
                            cv=5,
                            verbose=5, )

        grid.fit(X=X, y=y)
        logger.info(msg=f"Model fitted {self.model_name}")
        # select best model
        self.model = grid.best_estimator_
        # Store model
        self._store_model(obj=self)


class GenericModelConstructor(BaseTermFrequencyModel):
    """
    Generic Classification Model
    """

    @staticmethod
    def get_model_args(model: str) -> dict:
        """
        Return constructor arguments for the generic model
        :param model: model key to get its parameters
        :return:
        """
        return AVAILABLE_MODELS.get(model)

    def __init__(self, model_name: str, show_results: bool = False):
        _args = self.get_model_args(model=model_name)
        _args['show_results'] = show_results
        _args['model_name'] = model_name
        BaseTermFrequencyModel.__init__(self, **_args)