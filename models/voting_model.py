""""
Voting Classifier Implementation - SkLearn lib
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from crawler.dataset_builder import generate_dataset_for_input
from mlxtend.classifier import EnsembleVoteClassifier
from models.base import ObjectStore, TermFrequencyClassifier as TfClassifier
from models.logger import get_logger

logger = get_logger(__file__)

__all__ = ['VotingClassifier']


class VotingClassifier(ObjectStore):
    """
    Voting Classifier Model -
    Classifies data among stack of other classifications
    """
    VOTING_MODEL = 'voting.models'

    def __init__(self, estimators: dict, path: str) -> None:
        """
        Initialize Voting Classifier Model
        :type estimators: dict: Dict of estimators to be fitted.
        Each estimator has its own parameters inside param_grid
        """
        super().__init__(store_path=path)
        self.estimators = estimators
        self.model = None
        self.tf_vector = None

    def _get_stored_model_data(self):
        """
        Reads stored models. Retrieves EnsembleVoteClassifier models
        and its tf-idf pre-trained object
        :return:
        """
        logger.info(msg="Reading stored models")
        class_model = self.read_model(model_only=False)
        self.model = class_model.model
        self.tf_vector = class_model.tf_vector
        return self

    def fit(self, X: any = None, y: any = None, refit: bool = False):
        """
        Fit Voting Model, using estimators provided
        :param X: Array like. Predictor Data
        :param y: Array Like. Label to pe predicted
        :param refit: Force models refit
        :return:
        """
        if not refit or (X is None and y is None):
            return self._get_stored_model_data()

        # Build a new models
        X, self.tf_vector = TfClassifier.vectorize_data(X=X)
        self.model = EnsembleVoteClassifier(
            clfs=self.estimators,
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
        # Fit models
        logger.info(msg="Fitting models")
        self.model.fit(X=X_train, y=Y_train)
        # store models
        self.store_model(obj=self)

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
