"""
Keras LSTM implementation for fake news detection
"""
import numpy as np
import tensorflow as tf
from keras import models, layers
from keras.layers import Input, Dense, Bidirectional, LSTM, Embedding
from sklearn.model_selection import train_test_split
from src.models.interfaces import ParameterCnn, ParameterLstm
from src.models.object_store import ObjectStore
from src.preprocess.transformer import CustomTransformer
from typing import List, Union
from src.logger.logging import get_logger

logger = get_logger(__file__)

# SET GPU AS DEFAULT
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


class KerasBaseClassifier:

    def __init__(self, parameters: Union[ParameterLstm, ParameterCnn]):
        self.store = ObjectStore(path=f"./{parameters.model_name}.model")
        self.params = parameters
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

    def _compile(self):
        """"""

    def fit(self,
            X: any,
            y: any,
            refit: bool = False,
            clean_data: bool = False) -> None:
        """
        Fit tf.keras Model
        :param refit: bool :: Force model refit
        :param X: Array like, Input
        :param y: Array like, Output
        :param clean_data: bool :: Force data cleansing
        Parameters
        ----------
        clean_data
        """
        if not refit or X is None or y is None:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__
            return
        if clean_data:
            X = CustomTransformer().fit_transform(X=X)

        max_words = self.params.max_features
        X, self.tokenizer = self.vectorize_data(
            X=X, pad_len=self.params.pad_len, max_words=max_words)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, test_size=0.1)

        # Compile model
        self._compile()
        # Fit models
        self.model.fit(
            X_train, y_train,
            epochs=self.params.epochs,
            validation_data=(X_test, y_test),
            batch_size=self.params.batch_size,
        )

        self.store.store_model(obj=self)

    def predict(self,
                X: Union[str, List[str]],
                clean_data: bool = True):
        """
        Performs model prediction
        Parameters
        ----------
        X: Union[str, List[str]] :: Input text
        clean_data: bool :: performs data cleansing

        Returns
        -------

        """
        if not hasattr(self, 'model') or not self.model:
            _ = self.store.read_model()
            self.__dict__ = _.__dict__

        if clean_data:
            X = CustomTransformer().fit_transform(X=X)

        X = self.tokenizer.transform(np.array(X))
        return (self.model.predict(X) > 0.5).astype("bool")


class LstmClassifier(KerasBaseClassifier):
    """
    Keras LSTM implementation to detect Fake News
    """

    def _compile(self) -> None:
        """
        Compile models according Input/Output layers below
        """
        # define the models
        logger.info(f"Compiling model: {self.params.model_name}")
        inputs = Input(shape=(None,), dtype="int32")
        x = Embedding(
            input_dim=self.params.max_features,
            output_dim=self.params.layer_1)(inputs)
        # Add 3 bidirectional LSTMs
        x = Bidirectional(LSTM(units=self.params.layer_1,
                               return_sequences=True))(x)
        x = Bidirectional(LSTM(units=self.params.layer_2,
                               return_sequences=True))(x)
        x = Bidirectional(LSTM(units=self.params.layer_3))(x)
        # Add a classifier
        outputs = Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
            metrics=['AUC'], )
        logger.info(f"Model compiled: {self.params.model_name}")
        model.summary()
        self.model = model


class CnnClassifier(KerasBaseClassifier):
    """
    Keras CNN implementation to detect Fake News
    """

    def _compile(self):
        """
        Compile models according Input/Output layers below
        """
        logger.info(f"Model: CNN")
        model = models.Sequential()
        # Each word will be mapped to a vector with size = 100
        # Each X input will have at maximum size = (pad_len) words
        # The vocabulary = number of different words => vocab_size
        model.add(Embedding(input_dim=self.params.max_features,
                            output_dim=self.params.transform_size,
                            input_length=self.params.pad_len))
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
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
            metrics=['AUC'],
        )
        logger.info(f"Model compiled: CNN")
        model.summary()
        self.model = model


__all__ = ['LstmClassifier', 'CnnClassifier']
