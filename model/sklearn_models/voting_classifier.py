""""
Voting Classifier Implementation - SkLearn lib
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from scrapper.manage_dataset import generate_dataset_for_input
from mlxtend.classifier import EnsembleVoteClassifier
from model.sklearn_models.base import BaseTermFrequency as Base
from model.sklearn_models.base import ObjectStore, GenericModelConstructor
from model.logger import get_logger
from . import MODELS_PATH

logger = get_logger(__file__)


class VotingClassifier(ObjectStore):
    """
    Voting Classifier Model -
    Classifies data among stack of other classifications
    """
    VOTING_MODEL = 'voting.model'
    VOTING_PATH = f'{MODELS_PATH}{VOTING_MODEL}'

    def __init__(self, estimators: dict) -> None:
        """
        Initialize Voting Classifier Model
        :type estimators: dict: Dict of estimators to be fitted.
        Each estimator has its own parameters inside param_grid
        """
        super().__init__(store_path=self.VOTING_PATH)
        self.estimators = estimators
        self.model = None
        self.tf_vector = None

    def _get_stored_model_data(self):
        """
        Reads stored model. Retrieves EnsembleVoteClassifier model
        and its tf-idf pre-trained object
        :return:
        """
        logger.info(msg="Reading stored model")
        class_model = self._read_model(model_only=False)
        self.model = class_model.model
        self.tf_vector = class_model.tf_vector
        return self

    def fit(self, X: any = None, y: any = None, refit: bool = False,
            fit_base_estimators: bool = False):
        """
        Fit Voting Model, using estimators provided
        :param X: Array like. Predictor Data
        :param y: Array Like. Label to pe predicted
        :param refit: Force model refit
        :param fit_base_estimators: Force refit base estimators
        :return:
        """
        if not refit or (X is None and y is None):
            return self._get_stored_model_data()

        # Manage estimators
        logger.info(msg="Fitting estimator models")
        estimators_ = []
        for estimator in self.estimators.keys():
            gm = GenericModelConstructor(model_name=estimator)
            gm.fit(X=X, y=y, force=fit_base_estimators)
            estimators_.append((estimator, gm.model))
        logger.info(msg="Estimator models fitted")

        # Build a new model
        X, self.tf_vector = Base.vectorize_data(X=X)
        self.model = EnsembleVoteClassifier(
            clfs=estimators_,
            voting='soft',
            fit_base_estimators=False,
            verbose=2
        )
        # split dataset
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42)
        # Fit model
        logger.info(msg="Fitting model")
        self.model.fit(X=X_train, y=Y_train)
        # store model
        self._store_model(obj=self)

    def predict(self, text_df: pd.DataFrame):
        """
        :parameter: text_df: Text list to be predicted
        Generates prediction, given a text
        """
        if not self.model:
            # Restore information stored
            self.fit()

        X = generate_dataset_for_input(df=text_df)
        X = self.tf_vector.transform(X)
        return self.model.predict(X=X)