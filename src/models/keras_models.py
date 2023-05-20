"""
Keras LSTM, CNN implementation for fake news detection
"""
import keras.backend
import numpy as np
import tensorflow as tf
from keras import models, layers
from keras.layers import Input, Dense, Bidirectional, LSTM, Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from src.models.interfaces import ParameterCnn, ParameterLstm
from src.models.object_store import ObjectStore
from src.preprocess.clean_transformer import CleanTextTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from src.preprocess.word2vec_transformer import TokenizerTransformer

from typing import List, Union
from src.logger.logging import get_logger

logger = get_logger(__file__)


class KerasBaseClassifier:

    def __init__(self, parameters: Union[ParameterLstm, ParameterCnn]):
        self.store = ObjectStore(path=f"./{parameters.model_name}.model")
        self.params = parameters
        self.history = None
        self.tokenizer = None
        self.model = None

    def _compile(self, *args, **kwargs):
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
            X = CleanTextTransformer().fit_transform(X=X)

        y = y.astype(int)
        # Fit on entire Corpus
        self.tokenizer = TokenizerTransformer()
        self.tokenizer.fit(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=True, random_state=42, stratify=y, test_size=0.2)
        # Transform train/test sets
        X_train = self.tokenizer.transform(X_train)
        X_test = self.tokenizer.transform(X_test)
        counts = y_train.value_counts()
        weight_false = 1 / counts.values.min()
        weight_pos = 1 / counts.values.max()

        # define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_tn', mode="max", patience=10, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_tp', mode="max", patience=10, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1),
        ]

        # create hyperparameters
        metrics = [
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
        # Compile model
        self.model = KerasClassifier(
            build_fn=self._compile,
            verbose=1,
            metrics=metrics,
            batch_size=256,
            epochs=100,
            validation_data=(X_test, y_test),
            optimizer=tf.optimizers.legacy.Adam(1e-4),
            class_weight={0: weight_false, 1: weight_pos},
            callbacks=callbacks,
            shuffle=True
        )

        logger.info(msg=f"{self.params.model_name} : FIT STARTED")
        self.model.fit(X_train, y_train)
        self.history = self.model.history  # store history
        logger.info(msg=f"{self.params.model_name} : FIT DONE")
        self.store.store_model(obj=self)
        y_predict = (self.model.predict(X_test) > 0.5).astype(int)
        logger.info(f"{self.params.model_name} : SCORE_TEST (ROC_AUC) => {roc_auc_score(y_test, y_predict)}")
        logger.info(f"{self.params.model_name} : SCORE_TEST (ACCURACY) => {accuracy_score(y_test, y_predict)}")

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
            X = CleanTextTransformer().fit_transform(X=X)

        X = self.tokenizer.transform(np.array(X))
        return (self.model.predict(X) > 0.5).astype("bool")


class LstmClassifier(KerasBaseClassifier):
    """
    Keras LSTM implementation to detect Fake News
    """

    def _compile(self, optimizer, metrics):
        """
        Compile models according Input/Output layers below
        """
        # define the models
        keras.backend.clear_session()
        model = models.Sequential()
        model.add(Input(shape=(None,), dtype="int32"))
        model.add(Embedding(input_dim=self.params.max_features, output_dim=self.params.layer_1))
        # Add 3 bidirectional LSTMs
        model.add(Bidirectional(LSTM(units=self.params.layer_1, return_sequences=True)))
        model.add(Bidirectional(LSTM(units=self.params.layer_2, return_sequences=True)))
        model.add(LSTM(units=self.params.layer_3))
        # Add a classifier
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=metrics
        )
        self.model = model
        return model


class CnnClassifier(KerasBaseClassifier):
    """
    Keras CNN implementation to detect Fake News
    """

    def _compile(self, optimizer, metrics):
        """
        Compile models according Input/Output layers below
        """
        keras.backend.clear_session()
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
            optimizer=optimizer,
            metrics=metrics,
        )
        self.model = model
        return model


__all__ = ['LstmClassifier', 'CnnClassifier']
