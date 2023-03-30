from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from typing import List


class TfIDF:

    def __init__(self, max_words=40000, k_best=10000):
        """
        TF-IDF implementation
        Parameters
        ----------
        max_words
        k_best
        """
        self.max_words = max_words
        self.is_fitted = False
        self.vect = TfidfVectorizer(
            max_features=self.max_words,
            ngram_range=(1, 1))
        self.best = SelectKBest(chi2, k=k_best)
        self._vect = None

    def transform(self, raw_documents: List[str]):
        """

        Parameters
        ----------
        raw_documents:

        Returns -> sparse Matrix
        -------

        """
        X = self.vect.transform(raw_documents=raw_documents)
        return self.best.transform(X=X).toarray()

    def fit(self, raw_documents: List[str], y) -> None:
        """
        Fit model to raw texts, considering p_val >= 0.95
        Parameters
        ----------
        raw_documents
        y
        """
        if self.is_fitted:
            return

        self.vect.fit(raw_documents=raw_documents)
        X = self.vect.transform(raw_documents=raw_documents)
        # selects k_best columns using chi2
        self.best.fit(X=X, y=y)
        self.is_fitted = True
