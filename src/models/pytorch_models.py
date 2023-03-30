import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.models.interfaces import Store

device = "cpu" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)


class CNNClassifier(nn.Module):
    def __init__(self, parameters, store: Store):
        """
        Convolutional Net
        Parameters
        ----------
        parameters: Parameter
        store: Store
        """
        super(CNNClassifier, self).__init__()
        # Store parameters
        _ = parameters
        self.tokenizer = None
        self.store = store
        self.store.set_path(path=f"./{_.model_name}.model")
        # Parameters regarding text preprocessing
        self.pad_len = _.pad_len
        self.max_features = _.max_features
        # Model layers definitions
        self.batch_size = _.batch_size
        self.out_size = _.out_size
        self.stride = _.stride
        # Optimizer, Criterion, L. Rate, epochs ...
        self.lr = _.learning_rate
        self.epochs = _.epochs
        self.criterion = nn.BCELoss().to(device=device)
        self.optimizer = None
        # Convolutional definitions
        # Net shape: 4 Convolution layers definition
        self.conv1 = nn.Conv1d(in_channels=self.pad_len,
                               out_channels=self.out_size,
                               kernel_size=2,
                               stride=self.stride).to(device=device)
        self.conv2 = nn.Conv1d(in_channels=self.pad_len,
                               out_channels=self.out_size,
                               kernel_size=3,
                               stride=self.stride).to(device=device)
        self.conv3 = nn.Conv1d(in_channels=self.pad_len,
                               out_channels=self.out_size,
                               kernel_size=4,
                               stride=self.stride).to(device=device)
        self.conv4 = nn.Conv1d(in_channels=self.pad_len,
                               out_channels=self.out_size,
                               kernel_size=5,
                               stride=self.stride).to(device=device)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=self.stride).to(device=device)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=self.stride).to(device=device)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=self.stride).to(device=device)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=self.stride).to(device=device)
        self.embedding = nn.Embedding(num_embeddings=self.max_features + 1,
                                      embedding_dim=self.pad_len,
                                      padding_idx=0).to(device=device)

    def forward(self, x):
        """
        Forward -> Apply model(X)
        Parameters
        ----------
        x

        Returns
        -------

        """
        x_ = self.embedding(x)
        x_ = x_.transpose(1, 2).contiguous()
        x_ = x_.to(device=device)
        x1 = self.pool1(torch.relu(self.conv1(x_)))
        x2 = self.pool2(torch.relu(self.conv2(x_)))
        x3 = self.pool3(torch.relu(self.conv3(x_)))
        x4 = self.pool4(torch.relu(self.conv4(x_)))
        out = torch.cat((x1, x2, x3, x4), dim=2)
        out = out.reshape(out.size(0), -1)
        out = nn.Linear(out.shape[1], 1, device=device)(out)
        out = nn.Dropout(0.25)(out)
        out = torch.sigmoid(out)
        if out.shape == (1, 1):
            return [float(out.squeeze())]
        return out.squeeze()

    @staticmethod
    def vectorize_data(X: any, pad_len: int = 1000, max_words: int = 30000) -> List[list]:
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

    def _evaluate(self, x_batch, y_batch):
        """
        Performs evaluation for each (x, y) batch
        Parameters
        ----------
        x_batch: Features with batch_size
        y_batch: Labels with batch_size

        Returns
        -------

        """
        # Set the model in evaluation mode
        self.eval()
        # Disable gradients for evaluation
        with torch.no_grad():
            y_hat = self(x_batch)
            y_hat = y_hat.detach().numpy()
        correct = sum(1 for y, y_pred in zip(y_batch, y_hat) if round(y_pred) == y)
        return correct

    def _execute_train(self, x_batch: torch.tensor,
                       y_batch: torch.tensor) -> float:
        """
        Performs training for each (x, y) batch
        Parameters
        ----------
        x_batch: Features with batch_size
        y_batch: Labels with batch_size

        Returns
        -------
        """
        x_batch = x_batch.to(device=device)
        y_batch = y_batch.to(device=device)
        # Set training mode
        self.train()
        # Clear gradients
        self.optimizer.zero_grad()
        # Get prediction
        y_hat = self(x_batch)
        # Find the Loss
        loss = self.criterion(y_hat, y_batch)
        # Calculate gradients
        loss.backward()
        # Update Weights
        self.optimizer.step()
        # returns loss
        return float(loss.item())

    def fit(self, X, y, refit=False):
        """
        Fit model given X=predictors, y=Label
        Parameters
        ----------
        X
        y
        refit: bool :: Force model train

        Returns
        -------

        """
        if not refit or X is None or y is None:
            return

        X, self.tokenizer = self.vectorize_data(
            X=X, pad_len=self.pad_len, max_words=self.max_features)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y, shuffle=True, random_state=42)
        # train dataset
        X_train = torch.tensor(X_train).long()
        y_train = torch.tensor(y_train.values, dtype=torch.float)
        # test dataset
        X_test = torch.tensor(X_test).long()
        y_test = torch.tensor(y_test.values, dtype=torch.float)

        # Initialize loaders
        loader_train = DataLoader(list(zip(X_train, y_train)),
                                  batch_size=self.batch_size)
        loader_test = DataLoader(list(zip(X_test, y_test)),
                                 batch_size=self.batch_size)

        # Optimizer and criterion
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Set model in training model
        for epoch in range(0, self.epochs):
            epoch_loss = 0.0
            for i, (x_batch, y_batch) in enumerate(tqdm(loader_train)):
                epoch_loss += self._execute_train(x_batch=x_batch, y_batch=y_batch)

            correct = 0
            for i, (x_batch, y_batch) in enumerate(loader_test):
                correct += self._evaluate(x_batch=x_batch, y_batch=y_batch)

            print(f"EPOCH_LOSS: {round(epoch_loss, 2)}")
            print(f"EPOCH_ACC: {round(correct / len(loader_test), 2)}")

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
            X = self.tokenizer.transform(X)
            y_hat = self(torch.tensor(X))
            if len(y_hat) == 1:
                return y_hat[0] > 0.5
            predictions = [x > 0.5 for x in y_hat.detach().numpy()]
            return predictions
