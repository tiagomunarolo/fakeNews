"""
Keras LSTM implementation for fake news detection
"""
import tensorflow as tf
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Embedding
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from model.sklearn_models.base import ObjectStore as Store
from typing import List


class KerasLstm(Store):
    """
    Keras LSTM implementation to detect Fake News
    """
    __name__ = 'LSTM'

    def __init__(self, store_path: str):
        super().__init__(store_path=store_path)
        self.model_name = self.__name__
        self.tokenizer = None
        self.model = None

    def _tokenize(self, X: any) -> List[list]:
        """
        Tokenize Corpus Dataset
        :param X: Array like
        :return:
        """
        if not self.tokenizer:
            self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X)
        return self.tokenizer.texts_to_sequences(X)

    def _compile_model(self, shape) -> None:
        """
        Compile model according to Sequencial Input and layers below
        :param shape:
        """
        # define the model
        # input [n x (max_len)]
        # n = number of rows in dataset, max_len = pad size of vector, default=2048
        model = tf.keras.Sequential([
            Input(name='inputs', shape=[shape[1]]),
            Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                      output_dim=128),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(64)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # compile model
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])
        model.summary()
        self.model = model

    def fit(self, X: any, y: any, max_len: int = 512, epochs: int = 10) -> None:
        """
        Fit tf.keras Model
        :param epochs:  int : number of epochs
        :param X: Array like
        :param y: Array like
        :param max_len: int
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42)

        sequences = self._tokenize(X=X_train)
        X_train = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

        self._compile_model(shape=X_train.shape, )
        # set callback
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                patience=3,
                verbose=False,
                restore_best_weights=True
            )
        ]
        # Fit model
        self.model.fit(X_train, y_train,
                       epochs=epochs,
                       validation_split=0.2,
                       batch_size=64,
                       callbacks=callbacks)

        self._store_model(obj=self)