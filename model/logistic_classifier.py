"""
Logistic Regression Model - sklearn implementation
"""
from model.base import Base
from . import BASE_PATH
from sklearn.linear_model import LogisticRegression

LOGISTIC_PATH = BASE_PATH + "/logistic.pyc"


class LogisticModel(LogisticRegression, Base):
    """
    Logistic Regression Model
    """

    def __init__(self, show_results=False):
        Base.__init__(self, show_results=show_results)
        LogisticRegression.__init__(self)

    def run_model(self, force=False) -> None:
        """
        Run LR model and output its results
        """
        super().run_model(
            LOGISTIC_PATH,
            LogisticRegression,
            force)

    @staticmethod
    def predict_output(text_data, *args) -> str:
        """
        Returns boolean prediction (Verdadeiro ou Falso)
        :param text_data: str - Text string to be predicted
        :param args: Optional
        """
        return Base.predict_output(
            text_data,
            LOGISTIC_PATH)