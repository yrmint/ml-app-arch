from typing import Union, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from copy import deepcopy

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
                # bias=False
            ),
            # torch.nn.BatchNorm2d(out_features),
            # torch.nn.InstanceNorm2d(out_features, affine=True),
            # torch.nn.GroupNorm(num_groups=8, num_channels=out_features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_features,
                out_features,
                kernel_size=3,
                padding=1,
                # bias=False
            ),
            # torch.nn.BatchNorm2d(out_features),
            # torch.nn.InstanceNorm2d(out_features, affine=True),
            # torch.nn.GroupNorm(num_groups=8, num_channels=out_features),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # torch.nn.Dropout(dropout)
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

        # final_size = 150 // (2 ** len(channels))
        # self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(channels[-1], 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
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
        X_val: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        channels: tuple[int] = (32, 64, 128),
        num_classes: int = 10,
        dropout: float = 0.25,
        lr: float = 1e-3,
        label_smoothing: float = 0.15,
        weight_decay: float = 2e-4,
        patience_threshold: int = 10,
        epoch_n: int = 10,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose_tqdm: bool = True
    ):
        self.X_val, self.y_val = X_val, y_val
        self.channels = channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.weight_decay = weight_decay
        self.epoch_n = epoch_n
        self.batch_size = batch_size
        self.device = device
        self.verbose_tqdm = verbose_tqdm

        self.genres = settings.GENRE_LABELS
        self.patience_threshold = patience_threshold
        self._patience_counter = 0
        self.best_val_score = float('-inf')
        self.best_model_state = None
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

        if self.X_val is not None:
            X_val = self._torch_cast(self.X_val, dtype=torch.float32)
            y_val = self._torch_cast(self.y_val, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        val_dataloader = DataLoader(
            dataset=TensorDataset(X_val),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self._model = CNN(channels=self.channels, num_classes=self.num_classes,
                          dropout=self.dropout).to(self.device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min"
            )
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
            )

        epochs_iter = tqdm(range(self.epoch_n)) \
            if self.verbose_tqdm \
            else range(self.epoch_n)

        for epoch in epochs_iter:
            val_score = 0.0
            running_loss = 0.0
            self._model.train()
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                y_pred = self._model(x_batch)
                loss = criterion(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x_batch.size(0)

            if self.X_val is not None:
                self._model.eval()
                with torch.no_grad():
                    # add .numpy() after .cpu() if doesnt work
                    y_pred_val = np.vstack(
                        [self._model(x_batch.to(self.device)).detach().cpu()
                         for (x_batch,) in val_dataloader])
                y_pred_val = self._torch_cast(y_pred_val, dtype=torch.float32)
                val_loss = criterion(y_pred_val, y_val).item()
                val_score = f1_score(
                    y_true=y_val.argmax(dim=1).numpy(),
                    y_pred=y_pred_val.argmax(dim=1),
                    average='macro')

                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.best_model_state = deepcopy(self._model.state_dict())
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

            scheduler.step(val_loss)

            if self.verbose_tqdm:
                epoch_loss = running_loss / len(dataset)
                epochs_iter.set_description(
                    f"Epoch {epoch+1}/{self.epoch_n}\
                    Loss: {epoch_loss:.4f} Val: {val_score:.4f}"
                )

            if self._patience_counter >= self.patience_threshold:
                if self.verbose_tqdm:
                    print(f"\nEarly Stopped at {epoch+1} epoch")
                    break

        if self.best_model_state is not None:
            self._model.load_state_dict(self.best_model_state)

        # требование ClassifierMixin
        self.classes_ = np.unique(y)
        return self

    def fine_tune(self, X, y, epochs: int = 5, lr: float = 1e-6,
                  freeze: bool = True):
        assert self._model is not None, "Trying to fine tune an unfitted model"

        # Optionally freeze conv layers
        if freeze:
            for param in self._model.features.parameters():
                param.requires_grad = False

        X_tensor = self._torch_cast(X, dtype=torch.float32)
        y_tensor = self._torch_cast(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=self.lr,
            weight_decay=lr * 0.5
        )

        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
            )

        epochs_iter = tqdm(range(epochs)) if self.verbose_tqdm \
            else range(epochs)

        self._model.train()
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
                    f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}"
                )

        # Unfreeze freezed layers
        if freeze:
            for param in self._model.features.parameters():
                param.requires_grad = True

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
