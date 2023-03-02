import os.path
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scrapper.manage_dataset import create_final_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import pandas as pd

FINAL_PATH = "/Users/tiagomunarolo/Desktop/fakenews/scrapper/csv_data/final_dataset.csv"
LOGISTIC_PATH = "/Users/tiagomunarolo/Desktop/fakenews/model/models/logistic.pyc"


class LogisticModel(LogisticRegression):
    """
    Logistic Regression Model
    """

    def __init__(self, data_path, show_results=True):
        """"""
        super(LogisticModel).__init__()
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

        data = pd.read_csv(self.data_path)
        self.X = data.TEXT
        self.Y = data.LABEL

    def _vectorize_data(self):
        """
        Vectorize dataset
        """
        vectorize_data = TfidfVectorizer()
        vectorize_data.fit(self.X)
        self.X = vectorize_data.transform(self.X)

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

    def _fit_model(self, force=False):
        """
        Fit Logistic Regression model
        """
        X_train, X_test, Y_train, Y_test = self._split_data()
        if not os.path.exists(LOGISTIC_PATH) or force:
            self.model = LogisticRegression()
            self.model.fit(X_train, Y_train)
            # Store model
            self._store_model()
        else:
            with open(LOGISTIC_PATH, 'rb') as file:
                self.model = pickle.load(file=file)
        # accuracy score on the training data
        if self.show_results:
            self._show_results(X_train, X_test, Y_train, Y_test)

    def _store_model(self):
        """
        Save model to ./models dir
        """
        with open(LOGISTIC_PATH, 'wb') as file:
            pickle.dump(obj=self.model, file=file)

    def _show_results(self, X_train, X_test, Y_train, Y_test):
        """

        :param X_train:
        :param X_test:
        :param Y_train:
        :param Y_test:
        """
        # accuracy score on the training data
        print("TRAINING DATA:")
        print("ACCURACY_SCORE: ", self.model.score(X=X_train, y=Y_train))
        print("F1_SCORE: ", f1_score(y_true=Y_train, y_pred=self.model.predict(X_train)))
        print("TEST DATA")
        print("ACCURACY_SCORE: ", self.model.score(X=X_test, y=Y_test))
        print("F1_SCORE: ", f1_score(y_true=Y_test, y_pred=self.model.predict(X_test)))

        cm1 = confusion_matrix(y_true=Y_train, y_pred=self.model.predict(X_train))
        cm2 = confusion_matrix(y_true=Y_test, y_pred=self.model.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=['Falso', 'Verdadeiro'])
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Falso', 'Verdadeiro'])

        disp.plot()
        disp2.plot()
        plt.show()

    def run_model(self, force=False):
        """
        Run LR model and output its results
        """
        self._read_dataset()
        self._vectorize_data()
        self._fit_model(force=force)