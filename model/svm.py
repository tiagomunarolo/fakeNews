"""
SVM Model - sklearn implementation
"""
from . import BASE_PATH
from model.base import Base
from sklearn.svm import SVC

SVM_PATH = BASE_PATH + "/svm.pyc"


class SVMModel(SVC, Base):
    """
    SVC Regression Model
    """

    def __init__(self, show_results=False):
        Base.__init__(self, show_results=show_results)
        SVC.__init__(self)

    def run_model(self, force=False):
        """
        Run LR model and output its results
        """
        super().run_model(
            SVM_PATH,
            SVC,
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
            SVM_PATH)