import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureTransformer(nn.Module):
    def __init__(self, feature_count, width, heads, layers):
        super().__init__()
        self.embedding = nn.Linear(1, width)
        self.feature_identity = nn.Parameter(torch.zeros(feature_count, width))
        nn.init.normal_(self.feature_identity, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            activation="gelu", dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.regression = nn.Linear(feature_count * width, 1)

    def forward(self, values):
        tokens = self.embedding(values.unsqueeze(-1)) + self.feature_identity
        return self.regression(self.encoder(tokens).flatten(1)).squeeze(-1)


class TransformerRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, width=32, heads=4, layers=2, epochs=100,
                 learning_rate=1e-3, batch_size=256, random_state=52, device="cpu"):
        self.width = width
        self.heads = heads
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        model = FeatureTransformer(
            self.n_features_in_, self.width, self.heads, self.layers
        ).to(self.device)
        samples = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32).reshape(-1),
        )
        batches = DataLoader(
            samples, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            for features, targets in batches:
                optimizer.zero_grad()
                loss = loss_fn(model(features.to(self.device)), targets.to(self.device))
                loss.backward()
                optimizer.step()
        self.model_ = model.cpu().eval()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        values = torch.as_tensor(X, dtype=torch.float32)
        with torch.inference_mode():
            predictions = [
                self.model_(batch).numpy()
                for batch in values.split(self.batch_size)
            ]
        return np.concatenate(predictions)
