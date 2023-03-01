import os.path
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scrapper.manage_dataset import create_final_dataset
from sklearn.metrics import accuracy_score
import pandas as pd

FINAL_PATH = "/Users/tiagomunarolo/Desktop/fakenews/scrapper/csv_data/final_dataset.csv"


class LogisticModel(LogisticRegression):
    """
    Logistic Regression Model
    """

    def __init__(self, data_path):
        """"""
        super(LogisticModel).__init__()
        self.data_path = data_path
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

    def _fit_model(self):
        """
        Fit LR model
        """
        X_train, X_test, Y_train, Y_test = self._split_data()
        model_lr = LogisticRegression()
        model_lr.fit(X_train, Y_train)
        LogisticRegression()
        # accuracy score on the training data
        X_train_prediction = model_lr.predict(X_train)
        X_test_prediction = model_lr.predict(X_test)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
        print("TRAINING DATA", training_data_accuracy)
        print("TEST DATA", test_data_accuracy)

    def run_model(self):
        """
        Run LR model and output its results
        """
        self._read_dataset()
        self._vectorize_data()
        self._fit_model()


LogisticModel(data_path=FINAL_PATH).run_model()