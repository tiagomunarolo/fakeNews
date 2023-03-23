"""
Keras LSTM implementation for fake news detection
"""
import tensorflow as tf
from keras.layers import Input, Dense, Bidirectional, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from src.models.interfaces import Store
from typing import List, Protocol
from src.logger.logging import get_logger

logger = get_logger(__file__)
gpu = len(tf.config.list_physical_devices('GPU')) > 0
logger.info("GPU is available" if gpu else "NOT AVAILABLE")


class Parameter(Protocol):
    # HyperParameters of generic model
    max_features: int  # max words in data dictionary
    pad_len: int
    layer_1: int
    layer_2: int
    layer_3: int
    epochs: int
    batch_size: int
    # model metadata
    model_name: str


def compile_model(parameter) -> tf.keras.Model:
    """
    Compile models according Input/Output layers below
    """
    # define the models
    logger.info("Compiling model")
    inputs = Input(shape=(None,), dtype="int32")
    x = Embedding(
        input_dim=parameter.max_features,
        output_dim=parameter.layer_1)(inputs)
    # Add 3 bidirectional LSTMs
    x = Bidirectional(LSTM(units=parameter.layer_1,
                           return_sequences=True))(x)
    x = Bidirectional(LSTM(units=parameter.layer_2,
                           return_sequences=True))(x)
    x = Bidirectional(LSTM(units=parameter.layer_3))(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    # compile models
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'], )
    logger.info("Model compiled")
    return model


class LstmClassifier:
    """
    Keras LSTM implementation to detect Fake News
    """

    def __init__(self, parameters: Parameter, store: Store):
        self.store = store
        self.params = parameters
        self.store.set_path(path=f"./{parameters.model_name}.model")
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
                num_words=self.params.max_features)
        self.tokenizer.fit_on_texts(X)
        logger.info("Tokenizing data for LSTM...completed")
        return self.tokenizer.texts_to_sequences(X)

    def fit(self, X: any, y: any, refit=False) -> None:
        """
        Fit tf.keras Model
        :param refit: bool :: Force model refit
        :param X: Array like, Input
        :param y: Array like, Output
        """
        if not refit or X is None or y is None:
            _ = self.store.read_model()
            self.__class__ = _.__class__
            self.__dict__ = _.__dict__
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, train_size=0.1)

        X_train = self._tokenize(X_train)
        X_test = self._tokenize(X_test)
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.params.pad_len)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=self.params.pad_len)

        self.model = compile_model(parameter=self.params)
        # Fit models
        self.model.fit(X_train, y_train,
                       epochs=self.params.epochs,
                       validation_data=(X_test, y_test),
                       batch_size=self.params.batch_size,
                       )

        self.store.store_model(obj=self)

    def predict(self, X):
        """
        Returns prediction {1= True, 0 = False/Fake News}
        :param X:
        :return:
        """
        logger.info(f"Predicting data: {X.shape}")
        if not self.model:
            _ = self.store.read_model()
            self.__class__ = _.__class__
            self.__dict__ = _.__dict__

        X = self._tokenize(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.params.pad_len)
        logger.info("Prediction completed")
        return (self.model.predict(X) > 0.5).astype("bool")


__all__ = ['LstmClassifier']
