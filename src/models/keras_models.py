"""
Keras LSTM implementation for fake news detection
"""
import tensorflow as tf
from keras.layers import Input, Dense, Bidirectional, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from src.models.sklearn_models import ObjectStore as Os
from typing import List
from dataclasses import dataclass
from src.logger.logging import get_logger

logger = get_logger(__file__)
gpu = len(tf.config.list_physical_devices('GPU')) > 0
logger.info("GPU is available" if gpu else "NOT AVAILABLE")

__all__ = ['LstmClassifier']


@dataclass
class Parameter:
    # Model parameters
    max_features: int = 30000  # max words in data dictionary
    pad_len: int = 512
    layer_1: int = 256
    layer_2: int = 128
    layer_3: int = 56
    epochs: int = 10
    batch_size: int = 32

    # model metadata
    model_name: str = 'lstm'


class LstmClassifier(Os):
    """
    Keras LSTM implementation to detect Fake News
    """

    def __init__(self):
        super().__init__(path=f"./{Parameter.model_name}.model")
        self.model_name = Parameter.model_name
        self.tokenizer = None
        self.model = None

    def _tokenize(self, X: any) -> List[list]:
        """
        Tokenize Corpus Dataset
        :param X: Array like
        :return:
        """
        logger.info("Tokenizing data for LSTM")
        if not self.tokenizer:
            self.tokenizer = Tokenizer(
                num_words=Parameter.max_features)
        self.tokenizer.fit_on_texts(X)
        logger.info("Tokenizing data for LSTM...completed")
        return self.tokenizer.texts_to_sequences(X)

    def _compile_model(self) -> tf.keras.Model:
        """
        Compile models according Input/Output layers below
        """
        # define the models
        logger.info("Compiling model")
        inputs = Input(shape=(None,), dtype="int32")
        # Embed each integer in a 128-dimensional vector
        x = Embedding(
            input_dim=Parameter.max_features,
            output_dim=Parameter.layer_1)(inputs)
        # Add 3 bidirectional LSTMs
        x = Bidirectional(LSTM(units=Parameter.layer_1,
                               return_sequences=True))(x)
        x = Bidirectional(LSTM(units=Parameter.layer_2,
                               return_sequences=True))(x)
        x = Bidirectional(LSTM(units=Parameter.layer_3))(x)
        # Add a classifier
        outputs = Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)
        # compile models
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'],
                      verbose=0
                      )
        logger.info("Model compiled")
        return model

    def __get_model__(self):
        """
        Get stored model
        """
        _ = self.read_model()
        self.model = _.model
        self.tokenizer = _.tokenizer

    def fit(self, X: any, y: any, refit=False) -> None:
        """
        Fit tf.keras Model
        :param refit: bool :: Force model refit
        :param X: Array like, Input
        :param y: Array like, Output
        """
        if not refit:
            self.__get_model__()
            return self

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, train_size=0.1)

        X_train = self._tokenize(X_train)
        X_test = self._tokenize(X_test)
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=Parameter.pad_len)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=Parameter.pad_len)

        self.model = self._compile_model()
        # Fit models
        self.model.fit(X_train, y_train,
                       epochs=Parameter.epochs,
                       validation_data=(X_test, y_test),
                       batch_size=32,
                       )

        self.store_model(obj=self)

    def predict(self, X):
        """
        Returns prediction {1= True, 0 = False/Fake News}
        :param X:
        :return:
        """
        logger.info(f"Predicting data: {X.shape}")
        if not self.model:
            self.__get_model__()

        X = self._tokenize(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=Parameter.pad_len)
        logger.info("Prediction completed")
        return (self.model.predict(X) > 0.5).astype("bool")
