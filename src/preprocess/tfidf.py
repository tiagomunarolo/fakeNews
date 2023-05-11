from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from typing import List


class TokenizerTf:
    """
    keras.preprocessing.text.Tokenizer
    """

    def __init__(self, max_words: int = 20000, max_len: int = 300):
        """
        Parameters
        ----------
        max_words: int :: Number of words in vocabulary
        max_len: int :: max length of each sequence
        """
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
