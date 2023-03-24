import tensorflow as tf
import torch
import math
import torch.nn as nn
from typing import List
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.models.interfaces import Store
from torch.nn.functional import binary_cross_entropy


class CNNClassifier(nn.Module):
    def __init__(self, parameters, store: Store):
        super(CNNClassifier, self).__init__()
        params = parameters
        # Store parameters
        self.tokenizer = None
        self.store = store
        self.store.set_path(path=f"./{params.model_name}.model")
        # Parameters regarding text preprocessing
        self.seq_len = params.seq_len
        self.num_words = params.num_words
        self.embedding_size = params.embedding_size
        self.batch_size = params.batch_size
        self.lr = params.learning_rate
        self.epochs = params.epochs
        # Dropout definition
        self.dropout = nn.Dropout(0.25)
        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5
        # Output size for each convolution
        self.out_size = params.out_size
        # Number of strides for each convolution
        self.stride = params.stride
        # Embedding layer definition
        self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)
        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)
        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)
        # Fully connected layer definition
        self.fc = nn.Linear(self.in_features_fc(), 1)

    def in_features_fc(self):
        """Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        """
        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size

    def forward(self, x):
        x = self.embedding(x)
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        x2 = self.conv_2(x)
        x2 = torch.relu(x2)
        x2 = self.pool_2(x2)
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)
        out = self.fc(union)
        out = self.dropout(out)
        out = torch.sigmoid(out)
        return out.squeeze()

    def vectorize_data(self, X: any) -> List[list]:
        """
        Tokenize Corpus Dataset
        :param X: Array like
        :return:
        """
        if not self.tokenizer:
            self.tokenizer = Tokenizer(
                num_words=self.num_words)
        self.tokenizer.fit_on_texts(X)
        return self.tokenizer.texts_to_sequences(X)

    def fit(self, X, y, refit=False):

        if not refit or X is None or y is None:
            return

        X_train, _, y_train, _ = train_test_split(
            X, y,
            stratify=y, shuffle=True, random_state=42)

        X_train = self.vectorize_data(X_train)
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.seq_len)

        X_train = torch.tensor(X_train, )
        y_train = torch.tensor(y_train.values, )

        # Initialize loaders
        loader_train = DataLoader(list(zip(X_train, y_train)),
                                  batch_size=self.batch_size)

        # Optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # Set model in training model
        self.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (x, labels) in enumerate(loader_train):
                labels = labels.type(torch.FloatTensor)
                optimizer.zero_grad()
                outputs = self(x)
                loss = binary_cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i and i % 99 == 0:
                    print(f'[{epoch + 1}, '
                          f'{i + 1:5d}] loss: '
                          f'{running_loss / 100:.3f}')
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
            X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.seq_len)
            y_hat = self(torch.tensor(X))
            predictions = [x > 0.5 for x in y_hat.detach().numpy()]
            return predictions
