"""
Keras LSTM implementation for fake news detection
"""
import tensorflow as tf
from keras.layers import Input, Dense, Bidirectional, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from models.base import ObjectStore as Store
from typing import List

from models.logger import get_logger

logger = get_logger(__file__)
gpu = len(tf.config.list_physical_devices('GPU')) > 0
logger.info("GPU is available" if gpu else "NOT AVAILABLE")

__all__ = ['KerasLstm']


class KerasLstm(Store):
    """
    Keras LSTM implementation to detect Fake News
    """
    __MODEL_NAME__ = 'LSTM'
    __MAX_FEATURES__ = 30000  # max words in data dictionary
    __LSTM_LAYER_1__ = 256
    __LSTM_LAYER_2__ = 128
    __LSTM_LAYER_3__ = 56

    def __init__(self, store_path: str):
        super().__init__(store_path=store_path)
        self.model_name = self.__MODEL_NAME__
        self.tokenizer = None
        self.model = None
        self.max_len = 512

    def _tokenize(self, X: any) -> List[list]:
        """
        Tokenize Corpus Dataset
        :param X: Array like
        :return:
        """
        logger.info("Tokenizing data for LSTM")
        if not self.tokenizer:
            self.tokenizer = Tokenizer(
                num_words=self.__MAX_FEATURES__)
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
            input_dim=self.__MAX_FEATURES__,
            output_dim=self.__LSTM_LAYER_1__)(inputs)
        # Add 3 bidirectional LSTMs
        x = Bidirectional(LSTM(units=self.__LSTM_LAYER_1__,
                               return_sequences=True))(x)
        x = Bidirectional(LSTM(units=self.__LSTM_LAYER_2__,
                               return_sequences=True))(x)
        x = Bidirectional(LSTM(units=self.__LSTM_LAYER_3__))(x)
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

    def fit(self, X: any, y: any, max_len: int = 512, epochs: int = 10) -> None:
        """
        Fit tf.keras Model
        :param epochs:  int : number of epochs
        :param X: Array like, Input
        :param y: Array like, Output
        :param max_len: int
        """
        self.max_len = max_len
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, train_size=0.1)

        X_train = self._tokenize(X_train)
        X_test = self._tokenize(X_test)
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

        self.model = self._compile_model()
        # Fit models
        self.model.fit(X_train, y_train,
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       batch_size=512,
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
            _ = self.read_model(model_only=False)
            self.model = _.model
            self.tokenizer = _.tokenizer
            self.max_len = _.max_len

        X = self._tokenize(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_len)
        logger.info("Prediction completed")
        return (self.model.predict(X) > 0.5).astype("bool")
