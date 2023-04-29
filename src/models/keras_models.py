"""
Keras LSTM implementation for fake news detection
"""
import numpy as np
import tensorflow as tf
from keras import models, layers
from keras.layers import Input, Dense, Bidirectional, LSTM, Embedding
from sklearn.model_selection import train_test_split
from src.models.interfaces import Store
from typing import List, Protocol, Union
from src.logger.logging import get_logger

logger = get_logger(__file__)

# SET GPU AS DEFAULT
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


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


class Cnn(Protocol):
    # HyperParameters of generic model
    vocab_size: int  # max words in data dictionary
    pad_len: int
    epochs: int
    batch_size: int
    transform_size: int
    # model metadata
    model_name: str


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

    @staticmethod
    def vectorize_data(X: any, pad_len: int, max_words: int) -> List[list]:
        """

        Parameters
        ----------
        X
        pad_len
        max_words
        Returns
        -------

        """
        from src.preprocess.tfidf import TokenizerTf
        tf_vector = TokenizerTf(max_words=max_words,
                                max_len=pad_len)
        tf_vector.fit(raw_documents=X)
        return tf_vector.transform(raw_documents=X), tf_vector

    def _compile_(self, parameter: Parameter) -> None:
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
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
            metrics=['accuracy'], )
        logger.info("Model compiled")
        self.model = model

    def fit(self, X: any, y: any, refit: bool = False) -> None:
        """
        Fit tf.keras Model
        :param refit: bool :: Force model refit
        :param X: Array like, Input
        :param y: Array like, Output
        """
        if not refit or X is None or y is None:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__
            return

        X, self.tokenizer = self.vectorize_data(
            X=X, pad_len=self.params.pad_len, max_words=self.params.max_features)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, train_size=0.1)

        # Compile model
        self._compile_(parameter=self.params)
        # Fit models
        self.model.fit(X_train, y_train,
                       epochs=self.params.epochs,
                       validation_data=(X_test, y_test),
                       batch_size=self.params.batch_size,
                       )

        self.store.store_model(obj=self)

    def predict(self, X: Union[str, List[str]]):
        """
        Returns prediction {1= True, 0 = False/Fake News}
        :param X:
        :return:
        """
        if not self.model:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__

        X = np.array(X)
        sequences = self.tokenizer.transform(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=sequences, maxlen=self.params.pad_len)
        logger.info("Prediction completed")
        return (self.model.predict(X) > 0.5).astype("bool")


class CnnClassifier:
    def __init__(self, parameters: Cnn, store: Store):
        _ = parameters
        self.store = store
        self.store.set_path(path=f"./{_.model_name}.model")
        self.vocab_size = _.vocab_size
        self.transform_size = _.transform_size
        self.pad_len = _.pad_len
        self.epochs = _.epochs
        self.batch_size = _.batch_size
        self.tokenizer = None

    def _compile_(self):
        """
        Compile models according Input/Output layers below
        """
        model = models.Sequential()
        # Each word will be mapped to a vector with size = 100
        # Each X input will have at maximum size = (pad_len) words
        # The vocabulary = number of different words => vocab_size
        model.add(Embedding(input_dim=self.vocab_size,
                            output_dim=self.transform_size,
                            input_length=self.pad_len))
        model.add(layers.Conv1D(filters=512, kernel_size=7, activation='relu'))
        model.add(layers.MaxPooling1D(2, 2))
        model.add(layers.Conv1D(filters=256, kernel_size=5, activation='relu'))
        model.add(layers.MaxPooling1D(2, 2))
        model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling1D(2, 2))
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling1D(2, 2))
        model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling1D(2, 2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
            metrics=['accuracy'],
        )
        self.model = model

    @staticmethod
    def vectorize_data(X: any, pad_len: int, max_words: int) -> List[list]:
        """

        Parameters
        ----------
        X
        pad_len
        max_words
        Returns
        -------

        """
        from src.preprocess.tfidf import TokenizerTf
        tf_vector = TokenizerTf(max_words=max_words,
                                max_len=pad_len)
        tf_vector.fit(raw_documents=X)
        return tf_vector.transform(raw_documents=X), tf_vector

    def fit(self, X: any, y: any, refit: bool = False) -> None:
        """
        Fit tf.keras Model
        :param refit: bool :: Force model refit
        :param X: Array like, Input
        :param y: Array like, Output
        """
        if not refit or X is None or y is None:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__
            return

        X, self.tokenizer = self.vectorize_data(
            X=X, pad_len=self.pad_len, max_words=self.vocab_size)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, train_size=0.1)

        # Compile model
        self._compile_()
        # Fit models
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            validation_data=(X_test, y_test),
            batch_size=self.batch_size,
        )

        self.store.store_model(obj=self)

    def predict(self, X: Union[str, List[str]]):
        """
        Returns prediction {1= True, 0 = False/Fake News}
        :param X:
        :return:
        """
        if not self.model:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__

        X = np.array(X)
        sequences = self.tokenizer.transform(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=sequences, maxlen=self.pad_len
        )
        logger.info("Prediction completed")
        return (self.model.predict(X) > 0.5).astype("bool")


__all__ = ['LstmClassifier', 'CnnClassifier']
