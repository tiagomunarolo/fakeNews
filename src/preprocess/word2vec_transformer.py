from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf


class TokenizerTransformer(BaseEstimator, TransformerMixin):
    """
    keras.preprocessing.text.Tokenizer
    """

    def __init__(self):
        self.tokenizer = Tokenizer(num_words=40000)

    def transform(self, X, y=None):
        sequences = self.tokenizer.texts_to_sequences(texts=X)
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences=sequences, maxlen=300)

    def fit(self, X, *args, **kwargs) -> None:
        self.tokenizer.fit_on_texts(texts=X)
        return self
