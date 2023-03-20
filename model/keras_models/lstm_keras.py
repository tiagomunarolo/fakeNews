"""
Keras LSTM implementation for fake news detection
"""
import tensorflow as tf
from keras.layers import Input, Dense, Bidirectional, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from model.sklearn_models.base import ObjectStore as Store
from typing import List


class KerasLstm(Store):
    """
    Keras LSTM implementation to detect Fake News
    """
    __name__ = 'LSTM'
    __MAX_FEATURES__ = 30000  # max words in data dictionary
    __LSTM_LAYER_1__ = 512
    __LSTM_LAYER_2__ = 256
    __LSTM_LAYER_3__ = 128

    def __init__(self, store_path: str):
        super().__init__(store_path=store_path)
        self.model_name = self.__name__
        self.tokenizer = None
        self.model = None
        self.max_len = 512

    def _tokenize(self, X: any) -> List[list]:
        """
        Tokenize Corpus Dataset
        :param X: Array like
        :return:
        """
        if not self.tokenizer:
            self.tokenizer = Tokenizer(
                num_words=self.__MAX_FEATURES__)
        self.tokenizer.fit_on_texts(X)
        return self.tokenizer.texts_to_sequences(X)

    def _compile_model(self) -> tf.keras.Model:
        """
        Compile model according to Sequencial Input and layers below
        """
        # define the model
        # input [n x (max_len)]
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

        # compile model
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])
        model.summary()
        return model

    def fit(self, X: any, y: any, max_len: int = 512, epochs: int = 10) -> None:
        """
        Fit tf.keras Model
        :param epochs:  int : number of epochs
        :param X: Array like
        :param y: Array like
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
        # Fit model
        self.model.fit(X_train, y_train,
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       batch_size=64)

        self._store_model(obj=self)

    def predict(self, X):
        """
        Returns prediction {1= True, 0 = False/Fake News}
        :param X:
        :return:
        """
        if not self.model:
            _ = self._read_model(model_only=False)
            self.model = _.model
            self.tokenizer = _.tokenizer
            self.max_len = _.max_len

        X = self._tokenize(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_len)
        return (self.model.predict(X) > 0.5).astype("bool")