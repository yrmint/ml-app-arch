from typing import Union, List

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ml.core.config import settings


class CNN(torch.nn.Module):
    @staticmethod
    def _get_conv_block(
            in_features,
            out_features,
            dropout: float
    ) -> List[torch.nn.Module]:
        res_l = [
            torch.nn.Conv2d(
                in_features,
                out_features,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_features,
                out_features,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_features),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(dropout)
        ]

        return res_l

    def __init__(
            self,
            channels: tuple[int],
            num_classes: int,
            dropout: float
    ):
        super().__init__()
        assert channels is not None

        # Building a model: len(channels) of convolutional blocks + classifier

        self.features = torch.nn.Sequential(
            *[
                module
                for i in range(len(channels))
                for module in self._get_conv_block(
                    in_features=1 if i == 0 else channels[i - 1],
                    out_features=channels[i],
                    dropout=dropout
                )
            ],
        )

        # final_size = image_size // (2 ** len(channels))
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(channels[-1] * 4 * 4, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, X):
        X = self.features(X)
        X = self.adaptive_pool(X)
        X = self.classifier(X)
        return X


# SciKit wrapper for the model
class SciKitCNN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        channels: tuple[int] = (32, 64, 128, 256),
        num_classes: int = 10,
        dropout: float = 0.25,
        image_size: int = 150,
        lr: float = 1e-3,
        weight_decay: float = 2e-4,
        epoch_n: int = 10,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose_tqdm: bool = True
    ):
        self.channels = channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.image_size = image_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch_n = epoch_n
        self.batch_size = batch_size
        self.device = device
        self.verbose_tqdm = verbose_tqdm

        self.genres = settings.GENRE_LABELS
        self._model = None

    def _torch_cast(
            self,
            data: Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor],
            dtype=torch.float32
    ) -> torch.Tensor:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=dtype)
        return data

    def fit(self, X, y):
        X_tensor = self._torch_cast(X, dtype=torch.float32)
        y_tensor = self._torch_cast(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        self._model = CNN(channels=self.channels, num_classes=self.num_classes,
                          dropout=self.dropout, image_size=self.image_size
                          ).to(self.device)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.15)

        self._model.train()
        epochs_iter = tqdm(range(self.epoch_n)) \
            if self.verbose_tqdm \
            else range(self.epoch_n)

        for epoch in epochs_iter:
            running_loss = 0.0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                y_pred = self._model(x_batch)
                loss = criterion(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x_batch.size(0)

            if self.verbose_tqdm:
                epoch_loss = running_loss / len(dataset)
                epochs_iter.set_description(
                    f"Epoch {epoch+1}/{self.epoch_n} - Loss: {epoch_loss:.4f}"
                )

        # требование ClassifierMixin
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        # returns raw probas
        assert self._model is not None, "Trying to predict before fitting"

        X_tensor = self._torch_cast(X, dtype=torch.float32)
        pred_loader = DataLoader(
            dataset=TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

        self._model.eval()
        all_probas = []

        with torch.no_grad():
            for (x_batch,) in pred_loader:
                x_batch = x_batch.to(self.device)
                logits = self._model(x_batch)
                probas = torch.softmax(logits, dim=-1)
                all_probas.append(probas.cpu().numpy())

        return np.vstack(all_probas)

    def predict_genre(self, X) -> dict:
        # returns transformer-like dictionary (for inference)
        segment_probas = self.predict_proba(X)
        proba = np.mean(segment_probas, axis=0)

        result = [
            {"label": self.genres[i],
             "score": float(proba[i])}
            for i in range(len(self.genres))
        ]
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        return result

    def predict(self, X) -> np.ndarray:
        # returns 1 genre among all probas (for tests)
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def save_model(self, filename):
        torch.save(self._model, filename)

    def load_model(self, filename):
        device = torch.device(settings.DEVICE)
        # checkpoint = torch.load(filename)
        # self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model = torch.load(
            filename,
            weights_only=False,
            map_location=device
        )
        self._model.eval()
