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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from src.preprocess.word2vec_transformer import TokenizerTransformer

from typing import List, Union
from src.logger.logging import get_logger

logger = get_logger(__file__)


class KerasBaseClassifier:

    def __init__(self, parameters: Union[ParameterLstm, ParameterCnn]):
        self.store = ObjectStore(path=f"./{parameters.model_name}.model")
        self.params = parameters
        self.optimizer_weights = None  # Just in case
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

        self.tokenizer = TokenizerTransformer()
        X = self.tokenizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=True, random_state=42, stratify=y, test_size=0.2)

        # create hyperparameters
        optimizers = [tf.optimizers.legacy.Adam(),
                      tf.optimizers.legacy.SGD(),
                      tf.optimizers.legacy.RMSprop()
                      ]
        metrics = ['accuracy',
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   tf.keras.metrics.AUC()
                   ]
        # Compile model
        estimator = KerasClassifier(
            build_fn=self._compile,
            verbose=1,
            metrics=metrics,
            batch_size=32
        )

        param_grid = {
            "optimizer": optimizers,
            "metrics": [metrics],
            "epochs": [1, 3, 5, 10]
        }

        self.model = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=3,
            scoring=['roc_auc', 'f1'],
            refit='f1',
            n_jobs=1,
            verbose=5
        )

        logger.info(msg=f"{self.params.model_name} : FIT STARTED")
        self.model.fit(X_train, y_train)
        logger.info(msg=f"{self.params.model_name} : FIT DONE")
        # delete parameters to avoid pickle error
        self.optimizer_weights = self.model.estimator.sk_params['optimizer'].get_weights()
        del self.model.estimator.sk_params['optimizer']
        del self.model.estimator.sk_params['metrics']
        del self.model.best_estimator_.sk_params['optimizer']
        del self.model.best_estimator_.sk_params['metrics']
        self.store.store_model(obj=self)
        y_predict = self.model.predict(X_test)
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
