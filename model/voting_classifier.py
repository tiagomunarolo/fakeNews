""""
Voting Classifier Implementation - SkLearn lib
"""
import os
import pickle
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from model.base import GenericStoreModel
from . import FINAL_PATH, BASE_PATH as MODELS_PATH

PATH_DEFAULT = './classifier_models.csv'
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

    def _get_dataset(self, path=PATH_DEFAULT):
        """
        Reads dataset, otherwise create it
        X => prediction variables
        Y => Target
        X is composed by predictions of previous pre-trained models
        :param path: str
        """
        if not os.path.exists(path):
            """"
            If not exists yet, get original dataset to generate 
            prediction dataset. It's composed by text as input and 
            Boolean as a target
            """
            data = pd.read_csv(FINAL_PATH, index_col=0)
            columns = ['LABEL']
            columns += [str(x.model) for x in self.stored_models]
            df = pd.DataFrame(columns=columns)
            df['LABEL'] = data.LABEL
            for class_m in self.stored_models:
                tf_idf = class_m.tf_vector
                X = tf_idf.transform(data.TEXT)
                try:
                    prediction = class_m.model.predict(X)
                except TypeError:
                    prediction = class_m.model.predict(X.toarray())
                # store predictions to a new dataset
                df[str(class_m.model)] = prediction
            # Stores dataset
            df.to_csv(path_or_buf=path, index_label=False)

        data = pd.read_csv(path, index_col=0)
        # X is composed for predictions of all pre-trained models
        self.X = data.drop(columns=['LABEL'])
        # Y is the target variable (bool)
        self.Y = data.LABEL

    def fit_model(self, force=False):
        """
        Fit model
        """
        # Get stored models
        self._get_stored_models()
        # Get dataset
        self._get_dataset()
        # Manage estimators
        estimators = [(str(x.model), x.model) for x in self.stored_models]
        if not force:
            return self
        # Build a new model
        self.model = VotingClassifier(estimators=estimators)
        # Split dataset for training and test
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X,
            self.Y,
            test_size=0.2,
            stratify=self.Y,
            random_state=2)
        # Fit model
        self.model.fit(X_train, Y_train)
        # store model
        self._store_model(obj=self)

    def predict_output(self, text_data: str):
        """
        :parameter: text_data: str: Text to be predicted
        Generates prediction, given a text
        """
        if not self.stored_models:
            self._get_stored_models()
        if not self.model:
            self.model = self._read_model()

        columns = [str(x.model) for x in self.stored_models]
        df = pd.DataFrame(columns=columns)
        for m_class in self.stored_models:
            response = m_class.predict_output(
                text_data=text_data, model_=m_class)
            df[str(m_class.model)] = [response]
        response = self.model.predict(df)
        return True if response[0] else False