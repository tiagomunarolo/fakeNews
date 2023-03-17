""""
Voting Classifier Implementation - SkLearn lib
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from scrapper.manage_dataset import generate_dataset_for_input
from model.sklearn_models import GenericModelConstructor, AVAILABLE_MODELS
from mlxtend.classifier import EnsembleVoteClassifier
from . import DATASET_PATH, MODELS_PATH, GenericStoreModel, BaseVectorizeModel as Base

VOTING_MODEL = 'voting.model'
VOTING_PATH = f'{MODELS_PATH}{VOTING_MODEL}'


class VotingClassifierModel(GenericStoreModel):
    """
    Voting Classifier Model -
    Classifies data among stack of other classifications
    """

    def __init__(self):
        super().__init__(store_path=VOTING_PATH)
        self.stored_models = []
        self.model = None
        self.tf_vector = None
        self.X = None
        self.Y = None

    def _get_stored_models(self):
        """
        Read all saved models in self.stored_models variable
        All models are binary classes with its own pre-trained parameters
        """
        files = os.listdir(MODELS_PATH)
        for file_name in files:
            _file = MODELS_PATH + file_name
            if file_name == VOTING_MODEL or not _file.endswith('.model'):
                continue
            with open(_file, 'rb') as model_class:
                self.stored_models.append(pickle.load(file=model_class))

    def fit_model(self, force=False, fit_base_estimators=False):
        """
        Fit model
        """
        if not force:
            class_model = self._read_model(sk_model_only=False)
            self.model = class_model.model
            self.tf_vector = class_model.tf_vector
            return self
        # Manage estimators
        estimators_ = []
        for model_name in AVAILABLE_MODELS.keys():
            gm = GenericModelConstructor(model_name=model_name)
            gm.fit_model(force=fit_base_estimators)
            estimators_.append(gm.model)
        # Build a new model
        self.X, self.Y = Base.read_dataset(path=DATASET_PATH)
        self.X, self.tf_vector = Base.vectorize_data(X=self.X)
        self.model = EnsembleVoteClassifier(
            clfs=estimators_,
            fit_base_estimators=False
        )
        # split dataset
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X,
            self.Y,
            test_size=0.2,
            stratify=self.Y,
            random_state=42)
        # Fit model
        self.model.fit(X=X_train, y=Y_train)
        # clean training data used
        self.X, self.Y = (None, None)
        # store model
        self._store_model(obj=self)

    def predict(self, text_df: pd.DataFrame):
        """
        :parameter: text_df: Text list to be predicted
        Generates prediction, given a text
        """
        if not self.stored_models:
            self._get_stored_models()
        if not self.model:
            # Restore information stored
            self.fit_model()

        X = generate_dataset_for_input(df=text_df)
        X = self.tf_vector.transform(X)
        return self.model.predict(X=X)