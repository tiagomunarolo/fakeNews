"""
Base Regression Model Class - For Generic objects
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
import pandas as pd
import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning)


class Base:
    """
    Logistic Regression Model
    """

    def __init__(self, data_path=FINAL_PATH, show_results=True):
        """"""
        self.show_results = show_results
        self.data_path = data_path
        self.model = None
        self.X = None
        self.Y = None

    def _read_dataset(self):
        """
        Open dataset, otherwise create it
        """
        if not os.path.exists(self.data_path):
            create_final_dataset()
            time.sleep(3)

        data = pd.read_csv(self.data_path, index_col=0)
        description = data.TEXT.apply(lambda x: len(x.split())).describe()
        first_quartile = int(description['25%'])
        third_quartile = int(description['75%'])
        filter_df = data.TEXT.apply(lambda x: first_quartile <= len(x.split()) <= third_quartile)
        data = data[filter_df]
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

    def _fit_model(self, path: str, model, force: bool = False):
        """
        Fit Generic provided model
        :param path: model store path
        :param model: Generic Model (SVC, LinearRegression, etc)
        :param force: bool: Force fit of a new model
        """
        X_train, X_test, Y_train, Y_test = self._split_data()
        if not os.path.exists(path) or force:
            self.model = model()
            self.model.fit(X_train, Y_train)
            # Store model
            self._store_model(path=path)
        else:
            with open(path, 'rb') as file:
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
        Y_hat = self.model.predict(self.X)
        print(classification_report(y_true=self.Y, y_pred=Y_hat, target_names=NAMES_LABEL))

        cm1 = confusion_matrix(y_true=Y_train, y_pred=self.model.predict(X_train))
        cm2 = confusion_matrix(y_true=Y_test, y_pred=self.model.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=NAMES_LABEL)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=NAMES_LABEL)

        disp.plot()
        disp2.plot()
        plt.show(block=False)

    def run_model(self, *args):
        """
        Run LR model and output its results
        """
        self._read_dataset()
        self._vectorize_data()
        self._fit_model(
            path=args[0],
            model=args[1],
            force=args[2])

    @staticmethod
    def predict_output(text_data: str, *args):
        """
        :parameter: text_data: str
        Generates prediction, given a text
        """
        with open(args[0], 'rb') as file:
            m = pickle.load(file=file)
            if len(text_data.split()) <= 25:
                raise Exception("Texto muito curto, por favor, "
                                "prover no mínimo 25 palavras!")
            X = generate_dataset_for_input(text=text_data)
            X = m.tf_vector.transform(X)
            response = m.model.predict(X)
            if not response[0]:
                print("FALSO")
            else:
                print("VERDADEIRO")