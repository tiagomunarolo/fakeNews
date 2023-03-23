import tensorflow as tf
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class TextClassifier(nn.Module):

    def __init__(self, param):
        super(TextClassifier, self).__init__()
        # Parameters regarding text preprocessing
        self.seq_len: int = param.seq_len
        self.num_words: int = param.num_words
        self.embedding_size: int = param.embedding_size
        self.out_size: int = param.out_size
        self.stride: int = param.stride
        self.epochs: int = 2
        self.batch_size: int = param.batch_size
        self.learning_rate: float = param.learning_rate
        # Dropout definition
        self.dropout = nn.Dropout(0.25)
        # CNN parameters definition
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5
        # Output size for each convolution
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
        self.model = None

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
        # Sequence of tokes is filterd through an embedding layer
        x = self.embedding(x)
        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu(x2)
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        # Convolution layer 4 is applied
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        # Dropout is applied
        out = self.dropout(out)
        # Activation function is applied
        out = torch.sigmoid(out)
        return out.squeeze()

    def evaluation(self, loader_test):
        # Set the model in evaluation mode
        self.eval()
        predictions = []

        with torch.no_grad():
            for x_batch, y_batch in loader_test:
                y_pred = self(x_batch)
                predictions += list(y_pred.detach().numpy())
        return predictions

    def fit(self, X, y, refit=False):

        tokenizer = Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y, shuffle=True, random_state=42)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_train = tf.keras.preprocessing.sequence.pad_sequences(
            X_train, maxlen=self.seq_len)

        X_test = tokenizer.texts_to_sequences(X_test)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(
            X_test, maxlen=self.seq_len)

        X_train = torch.tensor(X_train, )
        X_test = torch.tensor(X_test, )
        y_train = torch.tensor(y_train.values, )
        y_test = torch.tensor(y_test.values, )

        # Initialize loaders
        loader_train = DataLoader(list(zip(X_train, y_train)), batch_size=self.batch_size)
        loader_test = DataLoader(list(zip(X_test, y_test)), batch_size=self.batch_size)

        # Define the loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            # Set model in training model
            self.train()
            predictions = []
            # Starts batch training
            for x_batch, y_batch in loader_train:
                y_batch = y_batch.type(torch.FloatTensor)
                # Feed the model
                y_pred = self(x_batch)
                # Loss calculation
                loss = F.binary_cross_entropy(y_pred, y_batch)
                # Clean gradientes
                optimizer.zero_grad()
                # Gradients calculation
                loss.backward()
                # Gradients update
                optimizer.step()
                # Save predictions
                predictions += list(y_pred.detach().numpy())

            # Evaluation phase
            test_predictions = self.evaluation(loader_test)
            # Metrics calculation
            bool_predictions = [x > 0.5 for x in predictions]
            train_accuary = accuracy_score(y_true=y_train, y_pred=bool_predictions)
            test_accuracy = accuracy_score(y_true=y_test, y_pred=test_predictions)
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (
                epoch + 1, loss.item(), train_accuary, test_accuracy))
