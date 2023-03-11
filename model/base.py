"""
Base Classification Model Class - For Generic objects
"""
import os.path
import matplotlib.pyplot as plt
import time
import pickle
from . import FINAL_PATH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scrapper.manage_dataset import create_final_dataset, generate_dataset_for_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning)


class Base:
    """
    Base Classifier Model
    """

    def __init__(self, **kwargs):
        """"""
        self.show_results = kwargs['show_results']
        self.data = kwargs.get('data', FINAL_PATH)
        self.model_type = kwargs['model_type']
        self.store_path = kwargs['store_path']
        self.param_grid = kwargs.get('param_grid', {})
        self.model = None
        self.X = None
        self.Y = None

    def _read_dataset(self):
        """
        Open dataset, otherwise create it
        """
        if not os.path.exists(self.data):
            create_final_dataset()
            time.sleep(3)

        data = pd.read_csv(self.data, index_col=0)
        # description = data.TEXT.apply(lambda x: len(x.split())).describe()
        # first_quartile = int(description['25%'])
        # third_quartile = int(description['75%'])
        # filter_df = data.TEXT.apply(lambda x: first_quartile <= len(x.split()) <= third_quartile)
        # data = data[filter_df]
        self.X = data.TEXT
        self.Y = data.LABEL

    def _vectorize_data(self, X=None):
        """
        Vectorize dataset
        """
        if not X:
            X = self.X
        self.tf_vector = TfidfVectorizer()
        self.tf_vector.fit(X)
        self.X = self.tf_vector.transform(X)

    def _split_data(self):
        """
        Split dataset into train/test data
        :return:
        """
        return train_test_split(
            self.X,
            self.Y,
            test_size=0.2,
            stratify=self.Y,
            random_state=2)

    def _fit_model(self, force: bool = False):
        """
        Fit Generic provided model
        :param force: bool: Force fit of a new model
        """
        X_train, X_test, Y_train, Y_test = self._split_data()
        if not os.path.exists(self.store_path) or force:
            grid = GridSearchCV(estimator=self.model_type(),
                                param_grid=self.param_grid)
            try:
                grid.fit(X_train, Y_train)
            except TypeError:
                grid.fit(X_train.toarray(), Y_train)
            # Store best model
            self.model = grid.best_estimator_
            self._store_model(path=self.store_path)
        else:
            with open(self.store_path, 'rb') as file:
                m = pickle.load(file=file)
                self.model = m.model
        # accuracy score on the training data
        if self.show_results:
            self._show_results(X_train, X_test, Y_train, Y_test)

    def _store_model(self, path):
        """
        Save model to ./models dir
        """
        with open(path, 'wb') as file:
            pickle.dump(obj=self, file=file)

    def _show_results(self, X_train, X_test, Y_train, Y_test):
        """
        Show classification report and Plot confusion Matrix
        :param X_train: X Train dataset
        :param X_test: X Test dataset
        :param Y_train: Y Train dataset
        :param Y_test: Y Test dataset
        """
        # accuracy score on the training data
        NAMES_LABEL = ['Falso', 'Verdadeiro']
        try:
            Y_hat = self.model.predict(self.X)
        except TypeError:
            X_train = X_train.toarray()
            X_test = X_test.toarray()
            Y_hat = self.model.predict(self.X.toarray())
        print("*" * 200)
        print(self.model)
        print(classification_report(y_true=self.Y, y_pred=Y_hat, target_names=NAMES_LABEL))
        print("*" * 200)

        cm1 = confusion_matrix(y_true=Y_train, y_pred=self.model.predict(X_train))
        cm2 = confusion_matrix(y_true=Y_test, y_pred=self.model.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=NAMES_LABEL)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=NAMES_LABEL)

        disp.plot()
        disp2.plot()

        plt.title(f"MODELO: {self.model}")
        plt.show(block=False)

    def run_model(self, force=False):
        """
        Run LR model and output its results
        """
        self._read_dataset()
        self._vectorize_data()
        self._fit_model(force=force)

    def predict_output(self, text_data: str):
        """
        :parameter: text_data: str
        Generates prediction, given a text
        """
        with open(self.store_path, 'rb') as file:
            m = pickle.load(file=file)
            X = generate_dataset_for_input(text=text_data)
            X = m.tf_vector.transform(X)
            try:
                response = m.model.predict(X)
            except TypeError:
                response = m.model.predict(X.toarray())
            if not response[0]:
                return False
            else:
                return True