"""Baseline models for protein function prediction.

All models follow the same interface:
    fit(X, Y) -> None
    predict(X) -> np.ndarray of confidence scores
    save(path) / load(path)
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy import sparse


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray | sparse.spmatrix) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)


class NaiveFrequency(BaseModel):
    """Predict training-set frequency for every protein (ignores embeddings)."""

    def __init__(self):
        self.term_frequencies = None

    def fit(self, X: np.ndarray, Y: np.ndarray | sparse.spmatrix) -> None:
        if sparse.issparse(Y):
            Y = Y.toarray()
        self.term_frequencies = Y.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.tile(self.term_frequencies, (n, 1))


class LightGBMMultilabel(BaseModel):
    """One LightGBM binary classifier per GO term. Embarrassingly parallel."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        n_jobs: int = -1,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
        }
        self.n_jobs = n_jobs
        self.classifiers = []
        self.n_terms = 0

    def fit(self, X: np.ndarray, Y: np.ndarray | sparse.spmatrix) -> None:
        import lightgbm as lgb
        from joblib import Parallel, delayed

        if sparse.issparse(Y):
            Y = Y.toarray()

        self.n_terms = Y.shape[1]

        def train_one(col_idx):
            y = Y[:, col_idx]
            # Skip if no positive examples (shouldn't happen with min_count filter)
            if y.sum() == 0:
                return None
            clf = lgb.LGBMClassifier(
                **self.params,
                n_jobs=1,
                verbose=-1,
                importance_type="gain",
            )
            clf.fit(X, y)
            return clf

        print(f"Training {self.n_terms} classifiers...")
        self.classifiers = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(train_one)(i) for i in range(self.n_terms)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((X.shape[0], self.n_terms), dtype=np.float32)
        for i, clf in enumerate(self.classifiers):
            if clf is not None:
                scores[:, i] = clf.predict_proba(X)[:, 1]
        return scores


class MLPMultilabel(BaseModel):
    """Multi-output MLP in PyTorch for multi-label classification."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 256,
        device: str | None = None,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [2048, 1024]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = None
        self.n_terms = 0

        if device is None:
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def _build_network(self, n_terms: int):
        import torch.nn as nn

        layers = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_terms))
        return nn.Sequential(*layers)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray | sparse.spmatrix,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | sparse.spmatrix | None = None,
        early_stopping_patience: int = 5,
    ) -> list[float]:
        """Train the MLP. Returns list of validation losses (or train losses if no val set)."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if sparse.issparse(Y):
            Y = Y.toarray()

        self.n_terms = Y.shape[1]
        self.net = self._build_network(self.n_terms).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        X_t = torch.FloatTensor(X)
        Y_t = torch.FloatTensor(Y)
        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation setup
        has_val = X_val is not None and Y_val is not None
        if has_val:
            if sparse.issparse(Y_val):
                Y_val = Y_val.toarray()
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            Y_val_t = torch.FloatTensor(Y_val).to(self.device)

        losses = []
        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.shape[0]

            epoch_loss /= len(dataset)

            # Validation
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(X_val_t)
                    val_loss = criterion(val_logits, Y_val_t).item()
                losses.append(val_loss)
                print(f"Epoch {epoch+1}/{self.epochs} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                losses.append(epoch_loss)
                print(f"Epoch {epoch+1}/{self.epochs} - loss: {epoch_loss:.4f}")

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        scores = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i : i + self.batch_size]
                logits = self.net(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                scores.append(probs)

        return np.concatenate(scores, axis=0)

    def save(self, path: str | Path):
        import torch

        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dims": self.hidden_dims,
                    "dropout": self.dropout,
                    "n_terms": self.n_terms,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None):
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = cls(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
            device=device,
        )
        model.n_terms = config["n_terms"]
        model.net = model._build_network(model.n_terms)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net = model.net.to(model.device)
        return model
