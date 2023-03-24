import tensorflow as tf
import torch
import torch.nn as nn
from typing import List
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.models.interfaces import Store
from torch.nn.functional import binary_cross_entropy

# TODO FIX ENTIRE CLASS

class CNNClassifier(nn.Module):

    def __init__(self, parameters, store: Store):
        super(CNNClassifier, self).__init__()
        # Parameters regarding text preprocessing
        self.tokenizer = None
        self.store = store
        self.path = f"./{parameters.model_name}.model"
        self.store.set_path(path=self.path)
        self.pad_len: int = parameters.pad_len
        self.max_features: int = parameters.max_features
        self.learning_rate: float = parameters.learning_rate
        self.batch_size = parameters.batch_size
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

    def vectorize_data(self, X: any) -> List[list]:
        """
        Tokenize Corpus Dataset
        :param X: Array like
        :return:
        """
        if not self.tokenizer:
            self.tokenizer = Tokenizer(
                num_words=self.max_features)
        self.tokenizer.fit_on_texts(X)
        return self.tokenizer.texts_to_sequences(X)

    def fit(self, X, y, refit=False):

        if not refit or X is None or y is None:
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y, shuffle=True, random_state=42)

        X_train = self.vectorize_data(X_train)
        X_test = self.vectorize_data(X_test)
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.pad_len)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=self.pad_len)

        X_train = torch.tensor(X_train, )
        y_train = torch.tensor(y_train.values, )

        # Initialize loaders
        loader_train = DataLoader(list(zip(X_train, y_train)),
                                  batch_size=self.batch_size)

        # Define the loss function and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        self.train()
        for epoch in range(1):
            # Set model in training model
            running_loss = 0.0
            for i, (x, labels) in enumerate(loader_train):
                labels = labels.type(torch.FloatTensor)
                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i and i % 100 == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        self.store.store_model(obj=self)

    def predict(self, X):
        """
        Returns prediction {1= True, 0 = False/Fake News}
        :param X:
        :return:
        """
        _ = self.store.read_model()
        self.__class__ = _.__class__
        self.__dict__ = _.__dict__
        with torch.no_grad():
            X = self.vectorize_data(X)
            X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.pad_len)
            X = torch.tensor(X)
            y_hat = self(X)
            return list(y_hat.detach().numpy())
