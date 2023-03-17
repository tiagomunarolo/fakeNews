""""
Voting Classifier Implementation - SkLearn lib
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from scrapper.manage_dataset import generate_dataset_for_input
from mlxtend.classifier import EnsembleVoteClassifier
from . import MODELS_PATH, GenericStoreModel, \
    BaseTermFrequencyModel as Base, AVAILABLE_MODELS, GenericModelConstructor
import logging

VOTING_MODEL = 'voting.model'
VOTING_PATH = f'{MODELS_PATH}{VOTING_MODEL}'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name="VOTING_MODEL")


class VotingClassifierModel(GenericStoreModel):
    """
    Voting Classifier Model -
    Classifies data among stack of other classifications
    """

    def __init__(self):
        super().__init__(store_path=VOTING_PATH)
        self.model = None
        self.tf_vector = None

    def fit(self, X: any, y: any, force=False, fit_base_estimators=False):
        """
        Fit model
        """
        if not force:
            logger.info(msg=f"FORCE = {force}, reading stored model")
            class_model = self._read_model(model_only=False)
            self.model = class_model.model
            self.tf_vector = class_model.tf_vector
            return self
        # Manage estimators
        logger.info(msg=f"FORCE = {force}, Fitting model")
        estimators_ = []
        for model_name in AVAILABLE_MODELS.keys():
            gm = GenericModelConstructor(model_name=model_name)
            gm.fit(X=X, y=y, force=fit_base_estimators)
            estimators_.append(gm.model)
        logger.info(msg="Estimator models fitted")

        # Build a new model
        X, self.tf_vector = Base.vectorize_data(X=X)
        self.model = EnsembleVoteClassifier(
            clfs=estimators_,
            voting='soft',
            fit_base_estimators=False
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
        logger.info(msg=f"Storing model data to {self.store_path}")
        self._store_model(obj=self)

    def predict(self, text_df: pd.DataFrame):
        """
        :parameter: text_df: Text list to be predicted
        Generates prediction, given a text
        """
        if not self.model:
            # Restore information stored
            self.fit(X=None, y=None)

        X = generate_dataset_for_input(df=text_df)
        X = self.tf_vector.transform(X)
        return self.model.predict(X=X)