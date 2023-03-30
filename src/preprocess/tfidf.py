from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from sklearn.feature_selection import chi2, SelectKBest
import tensorflow as tf
from typing import List


class TokenizerTf:
    """
    keras.preprocessing.text.Tokenizer
    """

    def __init__(self, max_words: int = 40000, max_len: int = 10000):
        self.is_fitted = False
        self.max_len = max_len
        self.tokenizer = Tokenizer(
            num_words=max_words)

    def transform(self, raw_documents: List[str]):
        """

        Parameters
        ----------
        raw_documents: List[str] :: list of raw texts

        Returns
        -------

        """
        if not self.is_fitted:
            self.fit(raw_documents=raw_documents)
        sequences = self.tokenizer.texts_to_sequences(texts=raw_documents)
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences=sequences, maxlen=self.max_len)

    def fit(self, raw_documents: List[str]) -> None:
        """

        Parameters
        ----------
        raw_documents: List[str] :: list of raw texts

        Returns
        -------

        """
        if self.is_fitted:
            return
        self.tokenizer.fit_on_texts(texts=raw_documents)
        self.is_fitted = True


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
